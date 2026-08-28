#!/usr/bin/env python3
"""Author the review-9 replication notebook (kept in the repo as a build script).

Regenerate with:  python3 code/scripts/make_review9_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"


def cell(kind: str, source: str, **extra) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    out = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == CODE:
        out.update({"execution_count": None, "outputs": []})
    out.update(extra)
    return out


INTRO = r"""# Review-9 replication runs — MovieLens-1M / Amazon / Gowalla

These are the runs the ninth review round asked for and that **cannot** be produced in the
manuscript workspace (it has no `data/ml-1m/ratings.dat` and no
`data/amazon-digital-music/interactions.csv`). Everything here writes into
`code/results/review9/`, which `scripts/make_review9_stats.py` already knows how to read:
**you only have to run the jobs and push the JSONs — the tables, the manifest and the
validation follow automatically.** No LaTeX editing is required to get the numbers into
both documents.

| Run | Reviewer issue | Output file | Published as | Typeset automatically? |
|---|---|---|---|---|
| `fixed-denominator` | Critical 1 (relative reweighting vs pure suppression) | `fixed_denominator_<ds>.json` | `tab:r9-fixed-denominator` + `-paired` | yes (Gowalla already in) |
| `utility-factorial` | Critical 4 (utility mismatch confounds H2) | `utility_factorial_<ds>.json` | `tab:r9-utility-factorial-replication` | yes |
| `prospective` | Critical 5 (target-conditioned games) | `prospective_<ds>.json` | `tab:r9-prospective-replication` | yes |
| `candidate-redraw` | High 8 (candidate-set dependence) | `candidate_redraw_<ds>.json` | `tab:r9-candidate-redraw` | yes |
| `stratified-null` | High 12 (exchangeability of the null) | `stratified_null_<ds>.json` | `tab:r9-stratified-null` | yes (Gowalla already in) |
| `compute-matched` | High 13 (equal-computation comparison) | `compute_matched_<ds>.json` | `tab:r9-compute-matched` | yes |
| `hardware` | High 16/17 (environment, drift) | `hardware_<ds>.json` | `tab:r9-hardware` | yes |

`<ds>` is `movielens`, `amazon` or `gowalla`. The MovieLens and Amazon instances are the
ones that need this machine; the Gowalla instances already run in the manuscript sandbox, so
the queue skips a `gowalla` output only when it parses as a complete run (a truncated file is re-run, not published). "Typeset automatically" means: after the run,
`make stats` regenerates `acmart-primary/tables/review9_benchmark_replications.tex` **and its
`actionshap-ipm` mirror byte-identically**, so there is no dangling reference and no manual
table to keep in sync. The prose in Supplementary Section S11 still needs a sentence per new
cohort — the last cell prints exactly which paragraphs, and the reply "see last commit" is
enough to have that done.

## Order of operations

1. **Cell 2** — set paths and the cohort sizes you can afford.
2. **Cell 3** — environment check (fails with a fix hint rather than losing hours).
3. **Cell 4** — pilot: every experiment on a handful of users, to verify the data,
   the keys and the real per-user cost; it rewrites the time estimates below.
4. **Cell 5** — launch the full queue detached (survives closing the browser tab).
5. **Cell 6** — poll whenever you come back.
6. **Cell 7** — rebuild tables + manifest + PDFs and validate.
7. **Cell 8** — commit/push the JSONs.

Runtimes measured on the manuscript sandbox (2 cores, ~3 GB RAM) are the basis of the
estimates in cell 2; scale them by `cores/2`. The pilot cell recomputes them on *your*
machine, which is the number to trust.
"""

CONFIG = r"""from pathlib import Path
import json, os, re, shlex, shutil, subprocess, sys, time

# ---------------------------------------------------------------- find the repo
def find_repo(start=None):
    here = Path(start or Path().resolve())
    for cand in [here, *here.parents]:
        if (cand / "paper-ideas" / "ActionShap" / "code" / "scripts"
                / "run_review9_experiments.py").exists():
            return cand / "paper-ideas" / "ActionShap"
        for sub in ("", "next-paper"):
            probe = cand / sub / "paper-ideas" / "ActionShap"
            if (probe / "code" / "scripts" / "run_review9_experiments.py").exists():
                return probe
    raise SystemExit("could not locate paper-ideas/ActionShap from "
                     f"{here}; set REPO_OVERRIDE below to the ActionShap directory")

REPO_OVERRIDE = None          # e.g. "/home/me/work/next-paper/paper-ideas/ActionShap"
PAPER   = Path(REPO_OVERRIDE) if REPO_OVERRIDE else find_repo()
CODE    = PAPER / "code"
SCRIPTS = CODE / "scripts"
RESULTS = CODE / "results" / "review9"
REPO_ROOT = PAPER.parents[1]   # the git checkout root, where the Makefile lives
# Scratch must stay OUTSIDE results/review9: the manifest hashes that directory
# recursively, so a pilot file inside it would silently change the frozen stamp.
SCRATCH = CODE / "results" / "_review9_scratch"

# ---------------------------------------------------------------- data locations
DATASET_PATHS = {
    "movielens": CODE / "data" / "ml-1m" / "ratings.dat",
    "amazon":    CODE / "data" / "amazon-digital-music" / "interactions.csv",
    "gowalla":   CODE / "data" / "gowalla" / "interactions.csv",
}
# Point these at wherever the files really live; they are passed as --ml-path etc.
DATASET_PATHS["movielens"] = Path(os.environ.get("AES_ML_PATH", DATASET_PATHS["movielens"]))
DATASET_PATHS["amazon"]    = Path(os.environ.get("AES_AMAZON_PATH", DATASET_PATHS["amazon"]))

# ---------------------------------------------------------------- the queue
# est_minutes are from the 2-core sandbox: fixed-denominator 600 users @ M=250 = 45 min.
# Set USERS per cohort; 1000 is the paper's primary cohort. Use 250-400 if the box is small.
PY = sys.executable   # every subprocess and `make` call below uses the kernel's interpreter

def find_texlive():
    '''Directory holding pdflatex/latexmk, even when the kernel's PATH lacks them.

    MacTeX lives in /Library/TeX/texbin, which a venv-launched Jupyter rarely
    inherits; without this probe the notebook told a reviewer with a working
    toolchain that none existed, and skipped the PDF rebuild that the stale-PDF
    guard is waiting on.
    '''
    import glob
    for name in ("pdflatex", "latexmk"):
        found = shutil.which(name)
        if found and (Path(found).parent / "pdflatex").exists():
            return str(Path(found).parent)
    candidates = ["/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin"]
    candidates += sorted(glob.glob("/usr/local/texlive/*/bin/*"))
    for cand in candidates:
        d = Path(cand)
        if (d / "pdflatex").exists() and (d / "latexmk").exists():
            return str(d)
    return None


TEX_BIN = find_texlive()

USERS = {"movielens": 1000, "amazon": 1000, "gowalla": 600}
PERM = 250                     # the primary diagnostic budget used everywhere in the paper
SCALE = 1.0                    # <1.0 shrinks every cohort and every estimate proportionally
ONLY = None                    # e.g. ["prospective", "hardware"] to run a subset

# `hardware` costs what it costs regardless of cohort size, so it is flagged fixed.
FIXED_COST = {"hardware"}

PLAN = [
    # (experiment, extra flags, why, minutes per 1000 users at PERM, or total if fixed)
    ("fixed-denominator", [], "Issue 1: normalized reweighting vs pure suppression", 75),
    ("prospective",       [], "Issue 5: games built from the model's own top-1", 90),
    ("stratified-null",   ["--r-null", "2000"], "Issue 12: recency/popularity-stratified nulls", 100),
    ("utility-factorial", [], "Issue 4: attribution-utility x outcome-utility 2x2", 190),
    ("compute-matched",   ["--mpair-grid", "250", "1000"], "Issue 13: equal scorer-call budgets", 230),
    ("candidate-redraw",  ["--redraws", "10"], "Issue 8: independent candidate sets", 380),
    ("hardware",          ["--timing-repeats", "5", "--timing-users", "50"], "Issues 16/17: environment + timings", 12),
]

def jobs():
    '''(experiment, dataset, argv, output path, est_minutes) for everything to run.'''
    out = []
    for experiment, extra, _why, minutes in PLAN:
        if ONLY and not any(k in experiment for k in ONLY):
            continue
        for dataset in ("movielens", "amazon", "gowalla"):
            data = DATASET_PATHS[dataset]
            if not data.exists():
                continue
            users = max(8, int(round(USERS[dataset] * SCALE)))
            args = [sys.executable, str(SCRIPTS / "run_review9_experiments.py"),
                    experiment, "--dataset", dataset,
                    "--ml-path", str(DATASET_PATHS["movielens"]),
                    "--amazon-path", str(DATASET_PATHS["amazon"]),
                    "--gowalla-path", str(DATASET_PATHS["gowalla"]),
                    "--users", str(users), "--permutations", str(PERM),
                    "--out", str(RESULTS), *extra]
            target = RESULTS / f"{experiment.replace('-', '_')}_{dataset}.json"
            scaled = minutes if experiment in FIXED_COST else minutes * users / 1000.0
            out.append((experiment, dataset, args, target, scaled))
    return out

print("paper dir :", PAPER)
print("repo root :", REPO_ROOT)
print("results   :", RESULTS)
print(f"planned jobs: {len(jobs())}")
for experiment, dataset, _a, _t, minutes in jobs():
    print(f"   {experiment:<20} {dataset:<10} ~{minutes:>5.0f} min")
print(f"total ~{sum(j[4] for j in jobs()) / 60:.1f} h at the sandbox reference speed "
      f"(2 cores); the pilot cell replaces these with times measured on this machine")
"""

ENV = r"""import platform, hashlib

print(f"python {platform.python_version()}  ({sys.executable})")
try:
    import matplotlib, numpy, pandas, pytest, scipy
    from sklearn.linear_model import Ridge
    print(f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
          f"scipy {scipy.__version__}, scikit-learn {Ridge.__module__.split('.')[0]} "
          f"{__import__('sklearn').__version__}, matplotlib {matplotlib.__version__}, "
          f"pytest {pytest.__version__}")
except Exception as exc:
    raise SystemExit(
        "install numpy/pandas/scipy/pytest into the interpreter running this kernel ("
        + str(sys.executable) + "). A managed environment refuses plain pip, so use "
        "the venv the repo documents: python3 -m venv .venv && .venv/bin/pip install "
        "numpy pandas scipy scikit-learn matplotlib pytest, then restart the kernel on "
        "the .venv your editor uses -- the collect cell runs the suite too, so pytest and "
        "scikit-learn have to be importable there.  (" + str(exc) + ")"
    )

cores = os.cpu_count() or 1
mem_gb = float("nan")
try:
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            mem_gb = int(line.split()[1]) / 1024 ** 2
            break
except OSError:
    pass
print(f"cores {cores}, ~{mem_gb:.0f} GB RAM" if mem_gb == mem_gb else f"cores {cores}")
free_gb = shutil.disk_usage(str(CODE)).free / 1024 ** 3
print(f"free disk in code/: {free_gb:.1f} GB  (need ~2 GB for the per-user JSONs)")

missing = []
for name, path in DATASET_PATHS.items():
    if path.exists():
        print(f"  [ok]      {name:<10} {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        missing.append(name)
        print(f"  [missing] {name:<10} {path}")
if missing:
    print("\nDatasets missing:", ", ".join(missing))
    print("Set AES_ML_PATH / AES_AMAZON_PATH or edit DATASET_PATHS in cell 2.")
    print("The queue skips absent datasets, so it will still run on what is present.")

for script in ("run_review9_experiments.py", "make_review9_stats.py", "make_result_manifest.py",
               "validate_manuscript.py"):
    assert (SCRIPTS / script).exists(), f"{SCRIPTS / script} missing -- wrong repo root?"
print("generator scripts present")
try:
    rev = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print("git revision:", rev or "(not a checkout)")
except Exception as exc:
    print("git unavailable:", exc)
print("pdflatex:", shutil.which("pdflatex") or (str(Path(TEX_BIN) / "pdflatex") + " (not on PATH; the Makefile and the rebuild cell use it anyway)") if TEX_BIN
      else "NOT FOUND in PATH or the usual MacTeX/texlive locations")
"""

REQUIRED = r"""# The keys scripts/make_review9_stats.py reads out of each run. A run that finishes
# without them produces no table, so this is checked in the pilot and after every run.
REQUIRED_KEYS = {
    "fixed_denominator": ["dataset", "records", "summary", "n_users_sampled", "paired"],
    "utility_factorial": ["dataset", "records", "summary"],
    "stratified_null":   ["dataset", "summary"],
    "prospective":       ["dataset", "users_total", "users_audited", "summary",
                          "covers_heldout_target_fraction"],
    "candidate_redraw":  ["dataset", "redraws", "between_redraw"],
    "compute_matched":   ["dataset", "curves"],
    "hardware":          ["dataset", "hardware", "peak_rss_mb", "timings_seconds"],
}
# keys inside one record, per experiment
RECORD_KEYS = {
    "fixed_denominator": ["user", "scorer", "aia_shapley_bounded", "aia_shapley_deletion",
                          "aia_lime_bounded", "aia_loo_bounded", "gap_shapley",
                          "signed_shapley_bounded", "mean_abs_effect"],
    "utility_factorial": ["user", "attr_utility", "outcome_utility", "aia_matched", "aia_cross"],
    "candidate_redraw":  [],          # summary-level payload
}


def inspect_run(path, experiment):
    '''Return (ok, notes) for one produced JSON against the generator's contract.'''
    notes = []
    if not Path(path).exists():
        return False, ["file missing"]
    payload = json.loads(Path(path).read_text())
    want = [k for k in REQUIRED_KEYS[experiment] if not (
        experiment == "fixed_denominator" and k == "paired" and payload.get("records"))]
    absent = [k for k in want if k not in payload]
    if absent:
        notes.append(f"missing top-level keys: {absent}")
    records = payload.get("records") or []
    if RECORD_KEYS.get(experiment):
        if not records:
            notes.append("no records")
        else:
            miss = {k for k in RECORD_KEYS[experiment] if k not in records[0]}
            if miss:
                notes.append(f"records missing {sorted(miss)}")
    if experiment in ("fixed_denominator", "utility_factorial"):
        defined = [r for r in records if r.get("aia_shapley_bounded") is not None
                   or r.get("aia_matched") is not None]
        if records and len(defined) < 0.05 * len(records):
            notes.append(f"only {len(defined)}/{len(records)} users yield a defined AIA")
    ok = not [n for n in notes if "missing" in n]
    return ok, notes


print("key contract defined")
"""

PILOT = r"""# ---------------------------------------------------------------- pilot
RESULTS.mkdir(parents=True, exist_ok=True)
PILOT_DIR = SCRATCH / "pilot"
PILOT_DIR.mkdir(parents=True, exist_ok=True)


def pilot_args(experiment, dataset, out_dir, users, perm, extra=()):
    '''Same flags as the queue, only fewer users.

    The first version of this cell shrank the expensive knobs (--redraws 2 instead of
    10, one grid point instead of two, --r-null 100 instead of 2000) to keep the pilot
    fast, and then extrapolated the result as if it had run the planned job. The
    measured numbers were therefore 5-20x too cheap for exactly the three jobs that
    dominate the queue, so the "measured on this machine" estimate was worse than the
    reference table it replaced. The pilot now runs the plan's own flags and pays a few
    minutes to be right; `hardware` is the single exception because its cost is fixed
    (it is reported as absolute minutes, never per 1000 users) and 50 timed users is
    pure waste on a 12-user cohort.
    '''
    args = [sys.executable, str(SCRIPTS / "run_review9_experiments.py"), experiment,
            "--dataset", dataset,
            "--ml-path", str(DATASET_PATHS["movielens"]),
            "--amazon-path", str(DATASET_PATHS["amazon"]),
            "--gowalla-path", str(DATASET_PATHS["gowalla"]),
            "--users", str(users), "--permutations", str(perm), "--out", str(out_dir)]
    args += list(extra)
    if experiment in FIXED_COST:
        args = [a for a in args if a not in ("--timing-users", "50")] + ["--timing-users", "10"]
    return args


pilot, MEASURED, FIXED_MIN = {}, {}, {}
for experiment, extra, why, _est in PLAN:
    if ONLY and not any(k in experiment for k in ONLY):
        continue
    dataset = next((d for d in ("gowalla", "amazon", "movielens")
                    if DATASET_PATHS[d].exists()), None)
    if dataset is None:
        print(f"skip {experiment}: no dataset present")
        continue
    t0 = time.time()
    proc = subprocess.run(pilot_args(experiment, dataset, PILOT_DIR, 12, PERM, extra),
                          cwd=str(CODE), capture_output=True, text=True)
    secs = time.time() - t0
    out = PILOT_DIR / f"{experiment.replace('-', '_')}_{dataset}.json"
    if proc.returncode != 0:
        print(f"[FAIL] {experiment:<20} {proc.stderr.strip()[-400:]}")
        pilot[experiment] = "FAILED"
        continue
    ok, notes = inspect_run(out, experiment.replace("-", "_"))
    # The pilot runs at the real budget, so this is a direct per-user extrapolation;
    # it slightly over-states the cohort cost because model fitting and candidate
    # construction are fixed cost amortised over only 12 users.
    per_1000 = secs / 12.0 * 1000.0 / 60.0
    if experiment in FIXED_COST:
        # A fixed-cost job's wall time does not grow with the cohort, so a per-1000-user
        # extrapolation of its 12-user runtime is meaningless (it would claim 28 min for
        # a 20 s job). Keep the absolute pilot minutes instead.
        FIXED_MIN[experiment] = secs / 60.0
        scale_note = f"fixed cost {FIXED_MIN[experiment]:4.1f} min, cohort-independent"
    else:
        MEASURED[experiment] = per_1000
        scale_note = f"~{per_1000:5.0f} min per 1000 users at M={PERM}"
    print(f"[{'ok ' if ok else 'WARN'}] {experiment:<20} {secs:6.1f} s   {scale_note}   "
          f"{'; '.join(notes)}")
    pilot[experiment] = "ok" if ok else "keys: " + "; ".join(notes)

print("\n== do the payloads reach the published tables? ==")
# The pilot is also a dry run of the *generator*: a run type whose table builder
# trips over the payload shape is a bug in make_review9_stats.py, and finding that
# out after a 10-hour queue means the hours bought nothing publishable.
_env = dict(os.environ, AES_REVIEW9_RESULTS=str(PILOT_DIR))
_gen = subprocess.run([PY, str(SCRIPTS / "make_review9_stats.py"), "--dry-run"],
                      cwd=str(CODE), capture_output=True, text=True, env=_env)
_found = [l for l in _gen.stdout.splitlines() if l.startswith("DRY_RUN_FLOATS:")]
rendered = json.loads(_found[0].split(": ", 1)[1]) if _found else []
need = {"fixed-denominator": ["tab:r9-fixed-denominator", "tab:r9-fixed-denominator-paired"],
        "prospective": ["tab:r9-prospective-replication"],
        "stratified-null": ["tab:r9-stratified-null"],
        "utility-factorial": ["tab:r9-utility-factorial-replication"],
        "candidate-redraw": ["tab:r9-candidate-redraw"],
        "compute-matched": ["tab:r9-compute-matched"],
        "hardware": ["tab:r9-hardware"]}
for experiment, labels in need.items():
    missing = [l for l in labels if l not in rendered]
    print(f"   {experiment:<20} {'-> renders ' + ', '.join(labels) if not missing else '!! NO FLOAT for ' + ', '.join(missing)}")
if _gen.returncode:
    print(_gen.stderr.strip()[-800:])
    pilot["generator"] = "make_review9_stats.py raised on a pilot payload"

print("\npilot status:", pilot)
if any(v != "ok" for v in pilot.values()):
    print("Resolve the flagged runs before launching the full queue: the published tables")
    print("are only written when the run's records carry the keys the generator reads.")
elif MEASURED:
    total = sum((FIXED_MIN.get(e, 0.0) if e in FIXED_COST
                 else MEASURED[e] * USERS[d] * SCALE / 1000.0)
                for e, _x, _w, _m in PLAN for d in USERS
                if DATASET_PATHS[d].exists() and not (ONLY and not any(k in e for k in ONLY)))
    print(f"Pilot passed. Measured queue: ~{total / 60:.1f} h for the cohorts in cell 2.")
    print("The queue cell now uses these measured costs for its estimates.")
"""

LAUNCH = r"""# ---------------------------------------------------------------- launch the queue
LOG = SCRATCH / "review9_runs.log"
RUNNER = SCRATCH / "run_review9_queue.sh"
SCRATCH.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

measured = MEASURED if "MEASURED" in globals() else {}
FIXED_MIN = FIXED_MIN if "FIXED_MIN" in globals() else {}
lines = ["#!/usr/bin/env bash",
         "# Generated by REVIEW9_REPLICATION_RUNS.ipynb -- resumable, skips finished runs.",
         "set -uo pipefail",
         f"cd {shlex.quote(str(CODE))}",
        # "finished" means "the JSON parses", not "a file is there": see write_json.
        "complete() { [ -f \"$1\" ] && " + shlex.quote(PY) +
        " -c 'import json,sys; json.load(open(sys.argv[1]))' \"$1\" >/dev/null 2>&1; }",
         'echo "=== queue start $(date -Is) ==="']
total, table = 0.0, []
for experiment, dataset, args, target, est in jobs():
    key = experiment.replace("-", "_")
    users = max(8, int(round(USERS[dataset] * SCALE)))
    if experiment in measured:                       # measured on this machine
        minutes = (FIXED_MIN.get(experiment, 0.0) if experiment in FIXED_COST
                   else measured[experiment] * users / 1000.0)
    else:                                            # est is already cohort-scaled
        minutes = est                                # jobs() applied USERS and SCALE
    total += minutes
    table.append((experiment, dataset, minutes, target))
    quoted = " ".join(shlex.quote(str(a)) for a in args)
    lines += [
        f'echo "--- $(date -Is) {experiment} {dataset}"',
        f'if complete {shlex.quote(str(target))}; then echo "skip (complete): {target.name}"; else',
        f"  {quoted} 2>&1 | tail -40 || echo \"FAILED {experiment} {dataset}\"",
        f'  echo "done $(date -Is) {experiment} {dataset} -> {target.name}"',
        "fi",
    ]
lines.append('echo "=== queue end $(date -Is) ==="')
RUNNER.write_text("\n".join(lines) + "\n")
RUNNER.chmod(0o755)

print(f"{len(table)} jobs, ~{total / 60:.1f} h wall clock if run one after another")
print("Measured from the pilot" if measured else "Estimated from the 2-core sandbox figures")
for experiment, dataset, minutes, target in table:
    print(f"   {experiment:<20} {dataset:<10} "
          f"{'exists' if target.exists() else f'~{minutes:>5.0f} min'}")

# Detached so it survives the notebook being closed; kill with the printed PID to stop.
log_handle = open(LOG, "ab", buffering=0)
RUN_PID = subprocess.Popen(["bash", str(RUNNER)], stdout=log_handle,
                           stderr=subprocess.STDOUT, start_new_session=True,
                           cwd=str(CODE)).pid
print(f"\nlaunched pid {RUN_PID}")
print(f"log:     {LOG}")
print(f"script:  {RUNNER}   (run it by hand any time: bash {RUNNER.name}; "
      f"each job calls the kernel's interpreter, not the shell's python3)")
print("Rerun this cell after an interruption: existing outputs are skipped.")
"""

STATUS = r"""# ---------------------------------------------------------------- status / polling
def queue_alive():
    if "RUN_PID" not in globals():
        return False
    try:
        os.kill(RUN_PID, 0)
    except OSError:
        return False
    return True


def tail(path=LOG, n=14):
    if not Path(path).exists():
        return "(no log yet)"
    return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])


def status():
    rows = []
    for experiment, dataset, args, target, est in jobs():
        name = target.name
        if target.exists():
            try:
                payload = json.loads(target.read_text())
            except Exception as exc:
                rows.append(f"  {experiment:<20} {dataset:<10} CORRUPT "
                            f"({exc.__class__.__name__}) -- the queue re-runs this one")
                continue
            users = len({r.get("user") for r in payload.get("records", [])})
            if not users:
                blocks = payload.get("summary") or {}
                first = next(iter(blocks.values()), {})
                users = first.get("n_users") if isinstance(first, dict) else None
            users = users or "n/a"
            ok, notes = inspect_run(target, experiment.replace("-", "_"))
            rows.append((experiment, dataset, "done" if ok else "done?", users,
                         f"{target.stat().st_size/1e3:.0f} kB", "; ".join(notes)))
        else:
            rows.append((experiment, dataset, "pending", "", "", ""))
    width = max(len(r[0]) + len(r[1]) for r in rows)
    for experiment, dataset, state, users, size, note in rows:
        print(f"  {experiment:<20} {dataset:<10} {state:<7} users={users!s:<5} {size:<8} {note}")
    done = sum(1 for r in rows if r[2].startswith("done"))
    print(f"\n{done}/{len(rows)} runs present in {RESULTS}")
    print("queue process:", "running" if queue_alive() else "not running (finished or stopped)")
    print("--- log tail ---")
    print(tail())


status()


def wait_for_finish(poll_seconds=60, max_minutes=240):
    '''Block until the queue process exits or the budget runs out.'''
    if "RUN_PID" not in globals():
        print("this notebook did not launch the queue; nothing to wait on")
        return
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline and queue_alive():
        time.sleep(poll_seconds)
    print("queue finished" if not queue_alive() else "still running after the wait budget")
    status()
"""

COLLECT = r"""# ---------------------------------------------------------------- rebuild + validate
def sh(args, cwd=REPO_ROOT, env=None):
    proc = subprocess.run([str(a) for a in args], cwd=str(cwd), capture_output=True,
                          text=True, env=env)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


print("== 1. regenerate every table from the matrices + these runs ==")
# the root target, not just the review-9 generator: `make stats` also refreshes the
# review-3/review-5 tables and re-freezes the manifest, which is what keeps the two
# documents typesetting the same numbers from the same files.
_rc, _out = sh(["make", "-C", str(REPO_ROOT), "stats", f"PY={PY}"])
_lines = [l for l in _out.splitlines()
          if l.startswith("wrote") or "ablation:" in l or "report" in l.lower()]
print("\n".join(_lines[-12:]) if _lines else _out[-800:])
if _rc:
    print(_out[-1500:])
    raise SystemExit("make stats failed; nothing downstream is trustworthy")

print("\n== 2. read the frozen manifest stamp ==")
stamp = json.loads((CODE / "results" / "manifest.json").read_text())["manifest_stamp"]
print("stamp:", stamp)

print("\n== 3. quote the new stamp in both documents ==")
pat = re.compile(r"(\\newcommand\{\\resultmanifeststamp\}\{)[0-9a-f]+(\})")
for name in ("acmmanuscript.tex", "supplementary.tex"):
    path = PAPER / "acmart-primary" / name
    text = path.read_text()
    if not pat.search(text):
        print(f"   !! {name}: no \\resultmanifeststamp macro -- add one")
        continue
    new = pat.sub(lambda m: m.group(1) + stamp + m.group(2), text)
    path.write_text(new)
    print(f"   {name}: -> {stamp}")

print("\n== 4. mirrors identical? ==")
for rel in ("review9_statistics.tex", "review9_benchmark_replications.tex",
            "appendix_s3b_effects.tex", "review5_validation.tex"):
    a, b = PAPER / "acmart-primary" / "tables" / rel, PAPER / "actionshap-ipm" / "tables" / rel
    same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
    print(f"   {rel:<40} {'identical' if same else 'DIFFERS (fix before pushing)'}")

print("\n== 5. validators ==")
for cmd in ([PY, str(SCRIPTS / "validate_manuscript.py")],
            [PY, str(SCRIPTS / "make_result_manifest.py"), "--check"],
            [PY, "-m", "pytest", "-q"]):
    rc, out = sh(cmd, cwd=CODE)
    print(" " + " ".join(cmd[1:]), "->", "PASS" if rc == 0 else "FAIL")
    if rc:
        print(out[-2500:])

print("\n== 6. rebuild the PDFs (skipped without a LaTeX toolchain) ==")
if TEX_BIN:
    rc, out = sh(["make", "-C", str(REPO_ROOT), "pdf"],
                 env=dict(os.environ, PATH=TEX_BIN + os.pathsep + os.environ.get("PATH", "")))
    print("make pdf (via %s/pdflatex) ->" % TEX_BIN, "PASS" if rc == 0 else "FAIL")
    if rc:
        print(out[-2500:])
    else:
        print("   PDFs rebuilt: the stale-PDF xfail guard should now pass on the next run.")
else:
    print("   no pdflatex/latexmk in PATH, /Library/TeX/texbin or /usr/local/texlive/*/bin/*.")
    print("   Install MacTeX (`brew install --cask mactex-no-gui`) or run")
    print("   `make -C %s pdf` where TeX lives; the PDFs stay stale until then." % REPO_ROOT)

print("\n== 7. what changed ==")
print(sh(["git", "status", "--short"], cwd=REPO_ROOT)[1][:2500])
"""

REPORT = r"""# ---------------------------------------------------------------- what still needs prose
# A float that renders but is cited by nobody is the mirror image of a dangling
# \ref: the numbers are in the PDF and nothing in the prose points at them. Once a
# run lands, the sentence that used to say "no claim is made" has to be upgraded.
_rendered = []
_tbl = PAPER / "acmart-primary" / "tables" / "review9_benchmark_replications.tex"
if _tbl.exists():
    _rendered = re.findall(r"\\label\{([^}]*)\}", _tbl.read_text())
_cited = ""
for _doc in ("supplementary.tex", "acmmanuscript.tex"):
    _dp = PAPER / "acmart-primary" / _doc
    if _dp.exists():
        _cited += _dp.read_text()
_orphan = [l for l in _rendered if f"ref{{{l}}}" not in _cited]
if _orphan:
    print("\nRendered tables no prose cites -- upgrade the S11 sentence that defers them:")
    for l in _orphan:
        print(f"   Table~\\ref{{{l}}}")
else:
    print("\nevery rendered replication float is cited by the prose")

labels = {
    "tab:r9-fixed-denominator": "Issue 1 (construct validity of the intervention)",
    "tab:r9-fixed-denominator-paired": "Issue 1 (paired user-level contrast)",
    "tab:r9-utility-factorial-replication": "Issue 4 (utility mismatch)",
    "tab:r9-stratified-null": "Issue 12 (stratified nulls)",
    "tab:r9-prospective-replication": "Issue 5 (target-conditioning)",
    "tab:r9-candidate-redraw": "Issue 8 (candidate-set resampling)",
    "tab:r9-compute-matched": "Issue 13 (equal computation)",
    "tab:r9-hardware": "Issues 16/17 (environment, drift)",
}
present = set()
for tex in (PAPER / "acmart-primary" / "tables").glob("*.tex"):
    present |= set(re.findall(r"\\label\{(tab:r9-[^}]*)\}", tex.read_text()))

print("tables now typeset from your runs:")
for label, issue in labels.items():
    print(f"   {'[x]' if label in present else '[ ]'} {label:<38} {issue}")

supp = (PAPER / "acmart-primary" / "supplementary.tex").read_text()
print("\nparagraphs in Supplementary S11 that quote numbers (they need the new cohort added):")
for para in re.findall(r"\\emph\{([^}]*)\}", supp):
    if "Issue" in para:
        print("   -", para.strip()[:78])
print('''
Then commit and push the branch and say "see last commit": the tables and the manifest
are already regenerated above, so what is left is prose -- the paragraphs above cite
MovieLens/Amazon numbers that must be read off the new JSONs, and the plan document's
rows for issues 5, 8 and 13 flip from RUN to FIXED.

    git -C <repo root> add -A paper-ideas/ActionShap
    git -C <repo root> commit -m "review-9: primary-cohort replications"
    git -C <repo root> push
''')
"""

FOOTER = r"""## Notes on cost, failures and honesty about what these runs can show

* **Costs** (2 cores, per 1 000 users, `M_pair=250`): `fixed-denominator` ≈ 75 min,
  `prospective` ≈ 90, `stratified-null` ≈ 100, `utility-factorial` ≈ 190 (it is a 2×2 of
  games), `compute-matched` ≈ 230 for a two-point grid, `candidate-redraw` ≈ 38 min per
  redraw (so 10 redraws on 1 000 users is a long run — `--redraws 5 --users 400` is a
  respectable first answer to Issue 8), `hardware` ≈ 12. Cell 4 replaces all of these
  with measured values on your machine.
* **Scratch stays out of the manifest**: pilot runs, the queue log and the
  generated `bash` queue live in `code/results/_review9_scratch/` (git-ignored), because
  `code/results/review9/` is hashed recursively by `make_result_manifest.py`.
* **Resumable by design**: a job whose output JSON exists is skipped. To redo one run,
  delete its file first. `SCALE = 0.4` in cell 2 shrinks every cohort if the box is small;
  `ONLY = ["prospective", "hardware"]` runs a subset. Nothing is overwritten silently.
* **Memory**: each run keeps per-user records, which is a few MB per 1 000 users except
  `utility-factorial` (4 cells) and `candidate-redraw` (per-user vectors). If the box has
  < 4 GB, run one experiment at a time: `ONLY = ["fixed-denominator"]`, then the next.
* **If a run produces a table full of `--`**: the AIA is undefined for users whose
  attribution game has no defined bounded AIA at `n_max = 20`. The `Nonzero`/`Users`
  columns in the generated tables exist to make that visible rather than hidden, and the
  pilot cell warns when fewer than 5 % of users yield a defined value. Reducing `--users`
  never fixes that; raising `--n-max` does, and it also changes the estimand — record it in
  the paragraph you write.
* **What these runs cannot show**: none of them retrains a recommender, so they bound the
  interpretation of the existing claims, they do not establish a competitive
  neural-recommender result (that remains Issue 2/11). `fixed-denominator` in particular
  showed on Gowalla that the *level and the cross-scorer ordering* of bounded AIA move with
  the denominator while the realized effect scale stays ~2·10⁻⁴; if the primary cohort
  disagrees, the paper must say so, and the tables are generated from the same contract so
  they will show it rather than hide it.
* **Do not hand-edit** `tables/review9_*.tex`: they are overwritten by
  `scripts/make_review9_stats.py`, which writes both mirrors. Edit the generator instead.
"""

cells = [
    cell(MD, INTRO),
    cell(CODE, CONFIG),
    cell(CODE, ENV),
    cell(CODE, REQUIRED),
    cell(CODE, PILOT),
    cell(CODE, LAUNCH),
    cell(CODE, STATUS),
    cell(CODE, COLLECT),
    cell(CODE, REPORT),
    cell(MD, FOOTER),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parents[2] / "notebooks" / "REVIEW9_REPLICATION_RUNS.ipynb"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print("wrote", target, f"({target.stat().st_size/1024:.1f} kB, {len(cells)} cells)")
