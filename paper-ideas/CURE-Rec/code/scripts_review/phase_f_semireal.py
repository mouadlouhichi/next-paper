"""Semi-real integration experiment (reviewer Phase 9, item 4).

Runs the exact intervention game twice on the same configuration and seeds:
once with the hand-crafted HistoryAwarePolicy base (the archived primary
design) and once with a BPR-MF ranker trained on simulator-logged feedback.
The comparison shows whether the CURE decision layer behaves consistently when
the base policy is a learned recommender.

Claim scope: simulator-conditional; the feedback loop is the disclosed CURE-Sim
mechanism. This does not create external causal evidence.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cure_rec.config import load_settings  # noqa: E402
from cure_rec.game import coalition_names  # noqa: E402
from cure_rec.observability import RunLogger  # noqa: E402
from cure_rec.pipeline import run_experiment  # noqa: E402
from cure_rec.planner import select_robust_portfolio  # noqa: E402
from cure_rec.semireal import (  # noqa: E402
    collect_interaction_log,
    fit_logged_bpr,
    learned_policy_factory,
)

ASSETS = ROOT / "results" / "reviewer_phase_assets" / "semireal_integration"
CONFIG = ROOT / "configs" / "curesim_full.yaml"
SEED = 42


def summarize(game, decision, tag: str) -> dict:
    robust = game.robust_improvements
    return {
        "base_policy": tag,
        "mode": decision.mode.value,
        "status": decision.status.value,
        "base_feasible": decision.base_feasible,
        "selected_mask": decision.selected_mask,
        "selected_portfolio": ";".join(decision.selected_interventions) or "abstain",
        "lower_improvement": decision.lower_improvement,
        "upper_improvement": decision.upper_improvement,
        "cost": decision.cost,
        "provider_disparity_upper": decision.provider_disparity_upper,
        "fatigue_upper": decision.fatigue_upper,
        "relevance_delta_lower": decision.relevance_delta_lower,
        "robust_phi": json.dumps({name: float(value) for name, value in game.robust_shapley.items()}),
        "grand_coalition_robust_value": float(robust[max(robust)]),  # placeholder replaced below
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    settings = load_settings(CONFIG)
    settings.run.seed = SEED
    started = time.time()

    # 1. Interaction log under the default base policy.
    log = collect_interaction_log(settings, seed=SEED)
    log.to_csv(ASSETS / "interaction_log_seed42.csv", index=False)
    click_rate = float(log["response"].mean())
    print(f"LOG: {len(log)} impressions, click rate {click_rate:.4f} ({time.time()-started:.0f}s)", flush=True)

    # 2. Learned ranker.
    model = fit_logged_bpr(log, factors=32, max_updates=60_000, seed=SEED)
    print(f"BPR fitted ({time.time()-started:.0f}s)", flush=True)

    rows = []

    # 3a. Reference game with the hand-crafted base policy.
    cfg_a = settings.model_copy(deep=True)
    cfg_a.run.name = "semireal-handcrafted"
    cfg_a.run.output_root = ASSETS / "runs" / "handcrafted"
    logger_a, game_a, decision_a = run_experiment(cfg_a)
    row_a = summarize(game_a, decision_a, "history_aware_handcrafted")
    row_a["grand_coalition_robust_value"] = float(game_a.robust_improvements[(1 << 6) - 1])
    rows.append(row_a)
    print(f"HANDCRAFTED: {row_a['selected_portfolio']} ({row_a['mode']}, {row_a['lower_improvement']:.5f}) ({time.time()-started:.0f}s)", flush=True)

    # 3b. Semi-real game with the learned base policy.
    cfg_b = settings.model_copy(deep=True)
    cfg_b.run.name = "semireal-learned-bpr"
    cfg_b.run.output_root = ASSETS / "runs" / "learned_bpr"
    logger_b = RunLogger(cfg_b)
    from cure_rec.game import run_exact_game
    from cure_rec.planner import build_explanation_card
    from cure_rec.reporting import emit_assets
    from cure_rec.planner import decision_to_dict

    try:
        game_b = run_exact_game(cfg_b, logger_b, policy_factory=learned_policy_factory(model))
        decision_b = select_robust_portfolio(game_b, cfg_b, logger_b)
        build_explanation_card(game_b, decision_b, logger_b)
        emit_assets(game_b, decision_b, cfg_b, logger_b)
        logger_b.write_json("artifacts/run_summary.json", {"decision": decision_to_dict(decision_b), "base_policy": "learned_bpr"})
        logger_b.close(status="completed")
    except Exception:
        logger_b.close(status="failed")
        raise
    row_b = summarize(game_b, decision_b, "learned_bpr_logged_feedback")
    row_b["grand_coalition_robust_value"] = float(game_b.robust_improvements[(1 << 6) - 1])
    rows.append(row_b)
    print(f"LEARNED BPR: {row_b['selected_portfolio']} ({row_b['mode']}, {row_b['lower_improvement']:.5f}) ({time.time()-started:.0f}s)", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(ASSETS / "semireal_comparison.csv", index=False)

    # Attribution comparison for the six players.
    attr_rows = []
    for name in game_a.robust_shapley:
        attr_rows.append({
            "intervention": name,
            "robust_phi_handcrafted": float(game_a.robust_shapley[name]),
            "robust_phi_learned_bpr": float(game_b.robust_shapley[name]),
        })
    pd.DataFrame(attr_rows).to_csv(ASSETS / "semireal_attribution_comparison.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "config_hash": settings.config_hash(),
        "impressions": int(len(log)),
        "click_rate": click_rate,
        "bpr": {"factors": 32, "max_updates": 60_000},
        "claim_scope": "simulator-conditional semi-real integration; not external causal evidence",
        "runtime_seconds": time.time() - started,
    }
    (ASSETS / "revision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("SEMIREAL INTEGRATION DONE", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
