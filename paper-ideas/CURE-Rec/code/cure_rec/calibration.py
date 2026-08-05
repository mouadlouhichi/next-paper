"""Pre-specified CURE-Sim behavioral calibration and phase-diagram sweeps.

This module evaluates *sensitivity of the disclosed simulator assumptions*; it
is not a post-hoc fitting routine and does not calibrate CURE-Sim to MovieLens.
Every design point uses the exact six-player game and independent environment
seeds.  The one-at-a-time (OAT) design makes individual assumptions legible,
while the Latin-hypercube design probes their joint interior efficiently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cure_rec.config import Settings
from cure_rec.experiments import SeedSweepResult, run_seed_sweep


CalibrationDesign = Literal["oat", "lhs"]


@dataclass(frozen=True)
class CalibrationPoint:
    """One pre-specified simulator/constraint configuration."""

    point_id: str
    values: dict[str, float | int]
    varied_parameter: str
    varied_value: float | int | None
    is_baseline: bool = False


@dataclass(frozen=True)
class CalibrationResult:
    """Small aggregate artifacts for a completed behavioral sensitivity study."""

    run_dir: Path
    configurations: pd.DataFrame
    seed_decisions: pd.DataFrame
    summary: pd.DataFrame
    attributions: pd.DataFrame
    interactions: pd.DataFrame


# Names intentionally describe scientific assumptions rather than implementation
# paths.  They are retained verbatim in manifests and result tables.
_PARAMETER_PATHS: dict[str, tuple[str, str]] = {
    "fatigue_strength": ("simulator", "fatigue_multiplier"),
    "repeat_threshold": ("simulator", "repeat_fatigue_threshold"),
    "horizon": ("simulator", "horizon"),
    "provider_threshold": ("constraints", "max_provider_disparity"),
    "provider_balance_strength": ("interventions", "provider_balance_weight"),
    "novelty_delayed_benefit": ("simulator", "novelty_preference_drift"),
    "exploration_cost": ("interventions", "costs.explore_slot"),
}


def _value(settings: Settings, parameter: str) -> float | int:
    parent_name, attribute = _PARAMETER_PATHS[parameter]
    parent = getattr(settings, parent_name)
    if attribute.startswith("costs."):
        return float(parent.costs[attribute.split(".", maxsplit=1)[1]])
    value = getattr(parent, attribute)
    return int(value) if isinstance(value, int) else float(value)


def _set_value(settings: Settings, parameter: str, value: float | int) -> None:
    parent_name, attribute = _PARAMETER_PATHS[parameter]
    parent = getattr(settings, parent_name)
    if attribute.startswith("costs."):
        parent.costs[attribute.split(".", maxsplit=1)[1]] = float(value)
    elif parameter in {"repeat_threshold", "horizon"}:
        setattr(parent, attribute, int(round(float(value))))
    else:
        setattr(parent, attribute, float(value))


def _slug(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4f}".replace("-", "m").replace(".", "p")


def default_oat_points(settings: Settings) -> list[CalibrationPoint]:
    """Return the fixed seven-assumption OAT design around a supplied baseline.

    The baseline is executed once. For every assumption we run two predeclared
    contrast values and use the baseline as the reference in the phase diagrams.
    This avoids silently spending seven duplicate baseline computations. The
    novelty assumption begins at zero, so its two contrasts are positive values.
    """
    baseline = {name: _value(settings, name) for name in _PARAMETER_PATHS}
    alternatives: dict[str, tuple[float | int, float | int]] = {
        "fatigue_strength": (0.75, 1.25),
        "repeat_threshold": (max(1, int(baseline["repeat_threshold"]) - 1), int(baseline["repeat_threshold"]) + 1),
        "horizon": (max(2, int(baseline["horizon"]) - 4), int(baseline["horizon"]) + 4),
        "provider_threshold": (
            max(0.01, float(baseline["provider_threshold"]) - 0.04),
            min(0.99, float(baseline["provider_threshold"]) + 0.04),
        ),
        "provider_balance_strength": (max(0.0, float(baseline["provider_balance_strength"]) - 0.20), float(baseline["provider_balance_strength"]) + 0.20),
        "novelty_delayed_benefit": (0.03, 0.06),
        "exploration_cost": (max(0.0, float(baseline["exploration_cost"]) - 0.05), float(baseline["exploration_cost"]) + 0.05),
    }
    points = [CalibrationPoint("baseline", baseline, "baseline", None, is_baseline=True)]
    for name, bounds in alternatives.items():
        for value in bounds:
            # If a caller supplied a baseline on an edge, evaluating it twice is
            # wasteful and does not provide independent information.
            if np.isclose(float(value), float(baseline[name])):
                continue
            values = dict(baseline)
            values[name] = value
            points.append(CalibrationPoint(
                point_id=f"oat-{name}-{_slug(value)}",
                values=values,
                varied_parameter=name,
                varied_value=value,
            ))
    return points


def latin_hypercube_points(settings: Settings, samples: int, *, seed: int = 20260805) -> list[CalibrationPoint]:
    """Create a reproducible space-filling joint-assumption design.

    Stratification is implemented locally to avoid adding SciPy as a runtime
    dependency.  Integer parameters are rounded after sampling and every sampled
    setting is recorded exactly in the configuration table.
    """
    if samples < 1:
        raise ValueError("Latin-hypercube calibration requires at least one sample")
    baseline = {name: _value(settings, name) for name in _PARAMETER_PATHS}
    bounds: dict[str, tuple[float, float]] = {
        "fatigue_strength": (0.70, 1.40),
        "repeat_threshold": (1.0, 5.0),
        "horizon": (4.0, 20.0),
        "provider_threshold": (0.20, 0.36),
        "provider_balance_strength": (0.0, 0.80),
        "novelty_delayed_benefit": (0.0, 0.08),
        "exploration_cost": (0.02, 0.20),
    }
    rng = np.random.default_rng(seed)
    unit = np.empty((samples, len(bounds)), dtype=float)
    for column in range(len(bounds)):
        unit[:, column] = (rng.permutation(samples) + rng.random(samples)) / samples
    points = [CalibrationPoint("baseline", baseline, "baseline", None, is_baseline=True)]
    for row_index, row in enumerate(unit, start=1):
        values: dict[str, float | int] = {}
        for (name, (lower, upper)), fraction in zip(bounds.items(), row, strict=True):
            candidate = lower + fraction * (upper - lower)
            values[name] = int(round(candidate)) if name in {"repeat_threshold", "horizon"} else float(candidate)
        points.append(CalibrationPoint(
            point_id=f"lhs-{row_index:03d}",
            values=values,
            varied_parameter="joint_lhs",
            varied_value=None,
        ))
    return points


def _settings_for_point(settings: Settings, point: CalibrationPoint, root: Path) -> Settings:
    configured = settings.model_copy(deep=True)
    for parameter, value in point.values.items():
        _set_value(configured, parameter, value)
    configured.run.name = f"calibration-{point.point_id}"
    configured.run.output_root = root / point.point_id
    return configured


def _decision_summary(decisions: pd.DataFrame, point: CalibrationPoint) -> dict:
    portfolios = decisions["selected_interventions"].astype(str)
    frequency = portfolios.value_counts(dropna=False)
    mode_portfolio = str(frequency.index[0])
    selected_repeat = portfolios.str.contains("repeat_cap", regex=False)
    return {
        "point_id": point.point_id,
        "varied_parameter": point.varied_parameter,
        "varied_value": point.varied_value,
        "is_baseline": point.is_baseline,
        "seed_count": int(len(decisions)),
        "selected_portfolio_mode": mode_portfolio,
        "selected_portfolio_mode_frequency": int(frequency.iloc[0]),
        "selection_stability": float(frequency.iloc[0] / len(decisions)),
        "repeat_cap_selection_rate": float(selected_repeat.mean()),
        "base_feasibility_rate": float(decisions["base_feasible"].astype(bool).mean()),
        "repair_rate": float((decisions["mode"] == "repair").mean()),
        "improvement_rate": float((decisions["mode"] == "improvement").mean()),
        "feasible_selection_rate": float(decisions["feasible"].astype(bool).mean()),
        "lower_improvement_mean": float(decisions["lower_improvement"].mean()),
        "lower_improvement_std": float(decisions["lower_improvement"].std(ddof=1)) if len(decisions) > 1 else 0.0,
        "provider_disparity_upper_mean": float(decisions["provider_disparity_upper"].mean()),
        "fatigue_upper_mean": float(decisions["fatigue_upper"].mean()),
    }


def _emit_calibration_figures(root: Path, summary: pd.DataFrame, attributions: pd.DataFrame, design: CalibrationDesign) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    baseline = summary[summary["is_baseline"]].iloc[0]
    parameters = [name for name in _PARAMETER_PATHS]

    if design == "oat":
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
        for axis, parameter in zip(axes.flat, parameters, strict=False):
            subset = summary[summary["varied_parameter"] == parameter].copy()
            reference = pd.DataFrame([{**baseline.to_dict(), "varied_parameter": parameter, "varied_value": _value_from_row(baseline, parameter)}])
            plot = pd.concat([subset, reference], ignore_index=True).sort_values("varied_value")
            axis.errorbar(plot["varied_value"], plot["lower_improvement_mean"], yerr=plot["lower_improvement_std"], marker="o", capsize=3, color="#2E86AB")
            axis.axhline(float(baseline["lower_improvement_mean"]), color="#999999", linestyle="--", linewidth=1)
            axis.set_title(parameter.replace("_", " "))
            axis.set_xlabel("assumption value")
            axis.set_ylabel("robust lower improvement")
        axes.flat[-1].axis("off")
        fig.suptitle("One-at-a-time CURE-Sim assumption sensitivity", y=1.02)
        fig.tight_layout()
        fig.savefig(figures / "calibration_figure_oat_lower_improvement.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        factors = pd.DataFrame([{
            "point_id": row["point_id"],
            **{parameter: _value_from_row(row, parameter) for parameter in parameters},
        } for _, row in summary.iterrows()])
        joined = summary.merge(factors, on="point_id", how="left", validate="one_to_one")
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
        for axis, parameter in zip(axes.flat, parameters, strict=False):
            axis.scatter(joined[parameter], joined["lower_improvement_mean"], c=joined["repair_rate"], cmap="viridis", vmin=0, vmax=1)
            axis.set_title(parameter.replace("_", " "))
            axis.set_xlabel("assumption value")
            axis.set_ylabel("robust lower improvement")
        axes.flat[-1].axis("off")
        fig.suptitle("Latin-hypercube CURE-Sim calibration: color = repair rate", y=1.02)
        fig.tight_layout()
        fig.savefig(figures / "calibration_figure_lhs_lower_improvement.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = summary.sort_values("point_id")
    x = np.arange(len(order))
    axes[0].bar(x, order["selection_stability"], color="#2E86AB")
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Portfolio-selection stability")
    axes[1].bar(x, order["repair_rate"], color="#E76F51", label="repair")
    axes[1].bar(x, order["base_feasibility_rate"], bottom=order["repair_rate"], color="#2A9D8F", label="base feasible")
    axes[1].set_ylim(0, 1.05); axes[1].set_title("Mode and base feasibility"); axes[1].legend()
    axes[2].bar(x, order["repeat_cap_selection_rate"], color="#6A4C93")
    axes[2].set_ylim(0, 1.05); axes[2].set_title("repeat_cap selection rate")
    for axis in axes:
        axis.set_xticks([])
        axis.set_xlabel("calibration configuration")
    fig.tight_layout()
    fig.savefig(figures / "calibration_figure_decision_stability.png", dpi=180)
    plt.close(fig)

    repeat = attributions[attributions["intervention"] == "repeat_cap"]
    if not repeat.empty:
        series = repeat.groupby("point_id", as_index=False).agg(phi_mean=("phi_mean", "mean"), phi_std=("phi_mean", "std"))
        series = series.set_index("point_id").reindex(order["point_id"]).reset_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(np.arange(len(series)), series["phi_mean"], yerr=series["phi_std"].fillna(0), marker="o", capsize=3, color="#6A4C93")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title("repeat_cap Shapley stability across calibration configurations")
        ax.set_xlabel("calibration configuration"); ax.set_ylabel("mean Shapley value")
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(figures / "calibration_figure_repeat_cap_shapley.png", dpi=180)
        plt.close(fig)


def _value_from_row(row: pd.Series, parameter: str) -> float | int:
    value = row.get(f"parameter_{parameter}")
    if pd.isna(value):
        raise ValueError(f"Calibration row does not contain {parameter}")
    return int(value) if parameter in {"repeat_threshold", "horizon"} else float(value)


def run_calibration_sweep(
    settings: Settings,
    seeds: Iterable[int],
    *,
    design: CalibrationDesign = "oat",
    lhs_samples: int = 16,
    lhs_seed: int = 20260805,
) -> CalibrationResult:
    """Run a pre-specified OAT or Latin-hypercube exact-game calibration sweep.

    This can be computationally expensive for a full CURE-Sim configuration.  It
    intentionally does no hyperparameter selection: every design point and seed
    is retained and summarized, including infeasible-base repair decisions.
    """
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one calibration seed is required")
    if design not in {"oat", "lhs"}:
        raise ValueError(f"Unknown calibration design: {design}")
    points = default_oat_points(settings) if design == "oat" else latin_hypercube_points(settings, lhs_samples, seed=lhs_seed)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(settings.run.output_root) / f"calibration-{design}-{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    configuration_rows: list[dict] = []
    decision_frames: list[pd.DataFrame] = []
    attribution_frames: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for point in points:
        configured = _settings_for_point(settings, point, root)
        result: SeedSweepResult = run_seed_sweep(configured, seed_list)
        configuration_rows.append({
            "point_id": point.point_id,
            "varied_parameter": point.varied_parameter,
            "varied_value": point.varied_value,
            "is_baseline": point.is_baseline,
            "config_hash": configured.config_hash(),
            "sweep_run_dir": str(result.run_dir),
            **{f"parameter_{name}": value for name, value in point.values.items()},
        })
        decisions = result.decisions.copy()
        decisions.insert(0, "point_id", point.point_id)
        decision_frames.append(decisions)
        attributions = result.attributions.copy()
        attributions.insert(0, "point_id", point.point_id)
        attribution_frames.append(attributions)
        interactions = result.interactions.copy()
        interactions.insert(0, "point_id", point.point_id)
        interaction_frames.append(interactions)
        summaries.append(_decision_summary(decisions, point))

    configurations = pd.DataFrame(configuration_rows)
    summary = pd.DataFrame(summaries).merge(
        configurations.drop(columns=["varied_parameter", "varied_value", "is_baseline", "sweep_run_dir"]),
        on="point_id", how="left", validate="one_to_one",
    )
    seed_decisions = pd.concat(decision_frames, ignore_index=True)
    attributions = pd.concat(attribution_frames, ignore_index=True)
    interactions = pd.concat(interaction_frames, ignore_index=True)
    configurations.to_csv(root / "calibration_configurations.csv", index=False)
    seed_decisions.to_csv(root / "calibration_seed_decisions.csv", index=False)
    summary.to_csv(root / "calibration_summary.csv", index=False)
    attributions.to_csv(root / "calibration_attributions.csv", index=False)
    interactions.to_csv(root / "calibration_interactions.csv", index=False)
    _emit_calibration_figures(root, summary, attributions, design)
    manifest = {
        "purpose": "Behavioral sensitivity analysis of disclosed CURE-Sim assumptions; not empirical calibration to an observational ratings dataset.",
        "design": design,
        "seeds": seed_list,
        "lhs_samples": lhs_samples if design == "lhs" else None,
        "lhs_seed": lhs_seed if design == "lhs" else None,
        "base_config_hash": settings.config_hash(),
        "parameters": list(_PARAMETER_PATHS),
        "points": [asdict(point) for point in points],
        "generated_files": sorted(path.name for path in root.iterdir() if path.is_file()),
    }
    (root / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CalibrationResult(root, configurations, seed_decisions, summary, attributions, interactions)
