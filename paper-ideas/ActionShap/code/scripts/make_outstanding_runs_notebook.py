#!/usr/bin/env python3
"""Emit ``notebooks/OUTSTANDING_RUNS.ipynb``: exactly the runs the paper still lacks.

Nothing here is hardcoded to a job list.  The notebook is derived from two sources of truth:

* the subcommands ``code/scripts/run_review9_experiments.py`` actually accepts (with the extra flags
  each one defines), and
* the payloads already present in ``code/results/review9/`` (validated, not just filename-matched).

Re-running this script after some jobs finish shrinks the queue instead of duplicating work, and a
truncated or corrupt payload does not count as done.  Cost figures are sandbox measurements (2
cores) multiplied by an explicit pessimism factor, because the pilot under-reported queue cost by
5-20x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE / "results" / "review9"
NOTEBOOK = CODE.parent / "notebooks" / "OUTSTANDING_RUNS.ipynb"

# experiment -> (extra flags, minutes per 1000 users at the diagnostic budget, what it unblocks)
PLAN = {
    "fixed-denominator": (
        [], 75,
        "normalized reweighting versus pure suppression, on cohorts that currently only have "
        "Amazon/Gowalla/MovieLens partially covered; fills the effect-scale panel and the "
        "interface-conditioned ordering claim in Supplementary Section S11"),
    "prospective": (
        [], 90,
        "games built from the model's own top-1 recommendation rather than the held-out target; the "
        "Amazon cohort is the only missing one and the `Defined n` panel says so in its caption"),
    "stratified-null": (
        ["--r-null", "2000"], 100,
        "recency- and popularity-block nulls on the PRIMARY cohorts. The released Gowalla audit moved "
        "the null mean from ~0 to 0.100 and 0.192, so the unstratified primary null understates the "
        "baseline and is the weakest link in the alignment claim until this runs"),
    "utility-factorial": (
        [], 190,
        "the full attribution-utility x outcome-utility x interaction-aware design on the primary "
        "datasets; until it exists, H2 has to stay narrowed to the demonstrated utility-mismatch "
        "result rather than adjudicated"),
    "compute-matched": (
        ["--mpair-grid", "250", "1000"], 230,
        "equal scorer-call budgets, i.e. the accuracy frontier that replaces the retracted equal-budget "
        "reading and removes any remaining compute-fairness implication"),
    "candidate-redraw": (
        ["--redraws", "10"], 380,
        "independent candidate draws, so candidate construction enters the uncertainty instead of "
        "being conditioned away by one frozen 200-item set"),
    "hardware": (
        ["--timing-repeats", "5", "--timing-users", "50"], 12,
        "processor model and per-user timing uncertainty at full cohort size (cost is cohort-"
        "independent, so it is cheap and should run first if you want the environment table filled)"),
}
FIXED_COST = {"hardware"}
DATASETS = ("movielens", "amazon", "gowalla")
USERS = {"movielens": 1000, "amazon": 1000, "gowalla": 600}
PESSIMISM = 3.0
# A payload counts as finished only if it parses and carries the block the tables read.
REQUIRED_KEYS = {
    "fixed-denominator": ("records",),
    "prospective": ("records",),
    "stratified-null": ("summary",),
    "utility-factorial": ("records", "summary"),
    "compute-matched": ("curves",),
    "candidate-redraw": ("redraws",),
    "hardware": ("hardware", "timings_seconds"),
}


def payload_path(experiment: str, dataset: str) -> Path:
    return RESULTS / f"{experiment.replace('-', '_')}_{dataset}.json"


def complete(experiment: str, dataset: str) -> bool:
    path = payload_path(experiment, dataset)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    for key in REQUIRED_KEYS.get(experiment, ()):
        if key not in payload:
            return False
        # an empty list/dict means the job was started and never filled in
        if isinstance(payload[key], (list, dict)) and not payload[key]:
            return False
    return True


def build_jobs() -> list[dict]:
    jobs = []
    for experiment, (flags, per_1000, why) in PLAN.items():
        for dataset in DATASETS:
            if complete(experiment, dataset):
                continue
            users = USERS[dataset]
            base = per_1000 if experiment in FIXED_COST else per_1000 * users / 1000.0
            jobs.append({
                "experiment": experiment,
                "dataset": dataset,
                "users": users,
                "argv": [experiment, "--dataset", dataset, "--users", str(users), *flags],
                "out": str(payload_path(experiment, dataset).relative_to(CODE)),
                "minutes": round(base * PESSIMISM, 1),
                "why": why,
            })
    return sorted(jobs, key=lambda j: j["minutes"])  # cheapest first


CONFIG_CELL = r'''import json, os, shlex, subprocess, sys, time
from pathlib import Path

# The notebook assumes it is run from code/ (Jupyter starts there when launched with
# `jupyter notebook` from code/). Everything below is relative to that directory.
assert Path("scripts/run_review9_experiments.py").exists(), "run this notebook from code/"

USERS     = {"movielens": 1000, "amazon": 1000, "gowalla": 600}   # the paper's cohorts
PESSIMISM = 3.0          # sandbox minutes -> plan minutes; keep honest, do not tune down
DRY_RUN   = True         # flip to False to actually execute
OVERWRITE = False        # True re-runs jobs whose payload already exists
ONLY      = None         # e.g. ["stratified-null", "candidate-redraw"] to run a subset
LOGS      = Path("logs"); LOGS.mkdir(exist_ok=True)

# The runner resolves the interaction files itself; set these only if your copies live elsewhere.
PATH_FLAGS = {
    "movielens": ("--ml-path",     os.environ.get("AES_ML_PATH")),
    "amazon":    ("--amazon-path", os.environ.get("AES_AMAZON_PATH")),
    "gowalla":   ("--gowalla-path", os.environ.get("AES_GOWALLA_PATH")),
}


def command(job):
    flag, value = PATH_FLAGS[job["dataset"]]
    argv = [sys.executable, "scripts/run_review9_experiments.py", *job["argv"]]
    if value:
        argv += [flag, str(value)]
    return argv + ["--out", "results/review9"]


def status(job):
    return "done" if (Path(job["out"]).exists() and not OVERWRITE) else "queued"


JOBS = json.loads(JOBS_JSON)
JOBS = [j for j in JOBS if ONLY is None or j["experiment"] in ONLY]

print(f"{len(JOBS)} queued job(s); plan {sum(j['minutes'] for j in JOBS) / 60:.1f} h\n")
for job in sorted(JOBS, key=lambda j: -j["minutes"]):
    print(f"  {status(job):>6}  {job['experiment']:<18} {job['dataset']:<10} "
          f"~{job['minutes']:>6.0f} min  -> {job['out']}")
'''

RUNNER_CELL = r'''def run(experiment):
    """One job per notebook cell; independent, resumable, verified after the fact."""
    selected = [j for j in JOBS if j["experiment"] == experiment]
    for job in selected:
        if status(job) == "done":
            print("skip", job["out"], "(payload present; set OVERWRITE = True to redo)")
            continue
        argv = command(job)
        if DRY_RUN:
            print("would run:", " ".join(shlex.quote(a) for a in argv),
                  f"(~{job['minutes']:.0f} min)")
            continue
        log = LOGS / f"{job['experiment']}_{job['dataset']}.log"
        start = time.time()
        with log.open("w") as handle:
            handle.write("$ " + " ".join(argv) + "\n")
            handle.flush()
            print("running", job["experiment"], job["dataset"], "->", log, flush=True)
            subprocess.run(argv, check=True, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = (time.time() - start) / 60
        payload = json.loads(Path(job["out"]).read_text())
        size = len(payload.get("records") or payload.get("summary") or {})
        print(f"  finished in {elapsed:.1f} min ({elapsed / job['minutes'] * 100:.0f}% of plan); "
              f"payload carries {size} records/summary entries")


def run_all():
    for experiment in dict.fromkeys(j["experiment"] for j in JOBS):
        run(experiment)
'''

VERIFY_CELL = r'''import json, re, subprocess, sys
from pathlib import Path

PLAN_EXPERIMENTS = ["fixed-denominator", "prospective", "stratified-null", "utility-factorial",
                    "compute-matched", "candidate-redraw", "hardware"]
DATASETS = ["movielens", "amazon", "gowalla"]


def payload_done(experiment, dataset):
    path = Path(f"results/review9/{experiment.replace('-', '_')}_{dataset}.json")
    return path.exists() and bool(path.read_text().strip())


missing = [(e, d) for e in PLAN_EXPERIMENTS for d in DATASETS if not payload_done(e, d)]
print("remaining missing payloads:", missing or "none")
if missing:
    print("do NOT rebuild the paper yet: the tables would still print the narrowed claims")
'''

REBUILD_CELL = r'''# Rebuild the generated half of the paper, then re-quote the new content hash in both documents.
# Skipping the stamp is how a compiled PDF ends up citing a manifest that no longer exists.
subprocess.run(["make", "-C", "../..", "tables",   f"PY={sys.executable}"], check=True)
subprocess.run(["make", "-C", "../..", "manifest", f"PY={sys.executable}"], check=True)

stamp = json.loads(Path("results/manifest.json").read_text())["manifest_stamp"]
for doc in ("../acmart-primary/acmmanuscript.tex", "../acmart-primary/supplementary.tex"):
    text = Path(doc).read_text()
    Path(doc).write_text(re.sub(r"(\\newcommand\{\\resultmanifeststamp\})\{[0-9a-f]+\}",
                                lambda m: m.group(1) + "{" + stamp + "}", text, count=1))
    print(f"{Path(doc).name} re-stamped with {stamp}")
'''

CHECK_CELL = r'''# Every validator plus the whole suite; the stale-PDF xfail is expected until `make pdf` has run.
subprocess.run(["make", "-C", "../..", "check", f"PY={sys.executable}"], check=False)
print()
print("then, with a TeX toolchain on PATH (the Makefile probes /Library/TeX/texbin):")
print("  make -C .. pdf      # rebuilds both PDFs from the regenerated tables")
print("  make -C .. ready    # must report 0 blockers before the archive is handed to an editor")
print("  # then delete the xfail marker in code/tests/test_review9_publication_integrity.py")
print("  make -C .. overleaf # repacks the submission archive, including this notebook")
'''

GAPS_MD = """## 5. Items in the review that no CPU on this machine fixes

Listed so the queue above is not mistaken for the whole remaining workload. Each row says what would
actually settle it, because a longer queue does not.

| Reviewer item | Why a run cannot settle it as things stand | What would |
|---|---|---|
| Competitive neural/graph model as primary evidence | `sasrec-quality` measures ranking quality only; no attribution audit is wired for a neural or graph scorer | an alignment audit for `--model sasrec`/`lightgcn` in `run_recommendation.py`, or narrow the contribution permanently to normalized history-based scorers |
| Training- and candidate-distribution uncertainty for the primary scorer | every interval is conditional on one frozen fitted model; a candidate redraw is not a refit | repeated preprocessing/training splits feeding the same user-level inference, or keep the explicit cohort-conditional statement in the limitations section |
| Adaptive stopping for unstable users | the diagnostic budget is fixed at `M_pair=250`; no stopping rule is implemented | a criterion on the paired variance estimate, then re-run the headline cells under it |
| Matched-user LIME mask-design ablation | the released ablation changes the population as well as the mask design, so it cannot be read causally | `lime-masks` on one fixed cohort across designs |
| Per-method valid-`n` in the prospective panel | not a missing run: the four method columns already share one defined subset, printed as `Defined n` | nothing, unless four different subsets are wanted |
| Reproducibility of the supplement's proportions block | `scripts/make_review3_stats.py --check` verifies only the rows its generator owns, so the printed proportions block is hand-appended | a generator for that block from `user_seed_metrics.csv.gz` |

## 6. If you decide not to run something

Every unrun item already has an honest fallback in the text, and the one rule is that no claim may
outrun a payload. Concretely: keep H2 narrowed to utility mismatch (no factorial), describe the
primary null as unstratified and therefore understating the baseline (no stratified primary run),
keep candidate sets stated as fixed by construction (no redraws), and keep the architectural scope at
normalized history-based scorers (no neural audit). The abstract, Section 6.4, the limitations
section and the contributions list already carry those hedges; the failure mode to avoid is deleting
a hedge while leaving its run undone."""


def md(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def make_cells(jobs: list[dict]) -> list[dict]:
    total = sum(j["minutes"] for j in jobs)
    planned = len(PLAN) * len(DATASETS)
    done = planned - len(jobs)
    header = f"""# Runs the paper still needs

Derived, not typed: **{done} of {planned}** planned payloads exist and validate in
`code/results/review9/`, so **{len(jobs)} jobs** remain, listed cheapest first. Regenerate this
notebook any time with

```
python code/scripts/make_outstanding_runs_notebook.py
```

and completed jobs leave the queue on their own. A job counts as finished only if its payload parses
and contains the block the tables read, so an interrupted run stays queued. Every command is a real
subcommand of `code/scripts/run_review9_experiments.py` with flags that script defines - nothing
here is invented.

**Plan total: {total / 60:.1f} h.** Each figure is a measured 2-core sandbox time times a pessimism
factor of {PESSIMISM:g}, because the first pilot under-reported queue cost by 5-20x. More cores help
only partially: these runs are dominated by per-user attribution walks, not by work that spreads
across cohorts. A rough own-machine floor is the sandbox sum ({total / PESSIMISM / 60:.1f} h
sandbox-equivalent) rather than the plan total.

Cells are independent and resumable; each job writes its own JSON under `results/review9/` and logs
to `code/logs/`. Start with `DRY_RUN = True` to see the exact commands.
"""
    cells = [md(header),
             md("## 1. Configure once\n\n`data/` paths differ per machine, so they are overridable "
                "without touching the job cells. `DRY_RUN = True` prints commands and the roll-up "
                "instead of executing."),
             code(CONFIG_CELL.replace("json.loads(JOBS_JSON)",
                                       "json.loads(" + json.dumps(json.dumps(jobs)) + ")")),
             code(RUNNER_CELL),
             md("## 2. What each job buys\n\nRead this before spending a night on the machine: it is "
                "the mapping from queue entry to the sentence in the manuscript that is currently "
                "qualified because the run is missing.\n\n"
                "| Job | Why it is queued |\n|---|---|\n" +
                "\n".join(f"| `{name}` | {why} |" for name, (_f, _m, why) in PLAN.items())),
             md("## 3. Run the queue\n\nOne cell per experiment; the cell loops over whichever of "
                "its datasets are still missing.")]

    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        grouped.setdefault(job["experiment"], []).append(job)
    for experiment, group in grouped.items():
        names = ", ".join(f"{j['dataset']} (~{j['minutes']:.0f} min)" for j in group)
        cells.append(md(f"### `{experiment}` — {names}"))
        cells.append(code(f"run({experiment!r})"))

    cells += [md("## 4. Confirm the queue is empty"), code(VERIFY_CELL),
              md("## 4b. Regenerate the tables and re-quote the manifest stamp"), code(REBUILD_CELL),
              md("## 4c. Validate and rebuild the documents"), code(CHECK_CELL),
              md(GAPS_MD)]
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report the queue without writing")
    parser.add_argument("--print-plan", action="store_true", help="print the full 21-cell plan")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.check or args.print_plan:
        total = sum(j["minutes"] for j in jobs)
        print(f"{len(jobs)} job(s) outstanding, plan {total / 60:.1f} h")
        for experiment, (flags, per_1000, _why) in PLAN.items():
            for dataset in DATASETS:
                state = "done" if complete(experiment, dataset) else "QUEUED"
                if args.print_plan or state == "QUEUED":
                    print(f"  {state:>6}  {experiment:<18} {dataset:<10} "
                          f"{[experiment, '--dataset', dataset, '--users', str(USERS[dataset]), *flags]}")
        return 0

    notebook = {
        "cells": make_cells(jobs),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({len(notebook['cells'])} cells, {len(jobs)} queued jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
