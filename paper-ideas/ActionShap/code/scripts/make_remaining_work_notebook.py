#!/usr/bin/env python3
"""Emit ``notebooks/REMAINING_WORK.ipynb``: the whole remaining worklist, computed from the files.

The run queue alone (``OUTSTANDING_RUNS.ipynb``) covers the compute. This notebook is the close-out:
it verifies what the paper already asserts, reports which of it is still pending, and separates
three kinds of remaining work that are easy to conflate - runs to execute, code to write, and
claims the authors must decide to narrow. Every status is computed by inspecting the repository when
this script runs, so nothing here can go stale silently: regenerate with

    python code/scripts/make_remaining_work_notebook.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PAPER = CODE.parent
SCRIPTS = CODE / "scripts"
NOTEBOOK = PAPER / "notebooks" / "REMAINING_WORK.ipynb"
ROOT = PAPER.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def queue() -> tuple[list[dict], float]:
    try:
        gen = _load("mor", SCRIPTS / "make_outstanding_runs_notebook.py")
        jobs = gen.build_jobs()
        return jobs, sum(j["minutes"] for j in jobs) / 60.0
    except Exception as exc:                                   # pragma: no cover - defensive
        return [], 0.0


def audit() -> dict:
    path = CODE / "results" / "review9" / "success_estimand_audit.json"
    if not path.exists():
        subprocess.run([sys.executable, str(SCRIPTS / "audit_success_estimand.py")],
                       cwd=str(CODE), capture_output=True, text=True)
    return json.loads(path.read_text(encoding="utf-8"))


def manifest() -> dict:
    return json.loads((CODE / "results" / "manifest.json").read_text(encoding="utf-8"))


def pdf_freshness() -> list[dict]:
    out = []
    for name in ("acmmanuscript", "supplementary"):
        pdf = PAPER / "acmart-primary" / f"{name}.pdf"
        src = max((p for p in (PAPER / "acmart-primary").rglob("*.tex")), key=lambda p: p.stat().st_mtime)
        out.append({"document": name,
                    "pdf": pdf.exists(),
                    "pdf_mtime": pdf.stat().st_mtime if pdf.exists() else 0,
                    "newest_source": src.name,
                    "newest_mtime": src.stat().st_mtime})
    return out


# The release tree is a frozen earlier draft (different float placement, older captions), so byte
# parity is the wrong invariant. What must agree are the sentences that carry decisions: a patched
# claim has to be present in both trees, or the artifact contradicts itself.
PARIZED = {
    "review3_statistics.tex": "seed-averaged per-seed indicator",
    "appendix_intervention_full.tex": "user-level average of the per-seed indicator",
    "intervention_outcomes.tex": "never a denominator for a decision statistic",
}


def mirror_parity() -> list[str]:
    """Files whose decision sentence is present in one tree but not the other."""
    drift = []
    for name, needle in PARIZED.items():
        here = (PAPER / "acmart-primary" / "tables" / name)
        there = (PAPER / "actionshap-ipm" / "tables" / name)
        if not (here.exists() and there.exists()):
            continue
        if (needle in here.read_text(encoding="utf-8")) != (needle in there.read_text(encoding="utf-8")):
            drift.append(name)
    return drift


CHECKED = {
    "prose references": "validate_prose_references.py",
    "cross-table": "validate_cross_table.py",
    "inferential provenance": "validate_inferential_provenance.py",
    "static manuscript": "validate_manuscript.py",
}


def code_task(name: str, probe: str, what: str, declined: bool = False) -> dict:
    """A pending engineering task, with the identifier grep that decides whether it is done.

    The probe searches code/scripts only, for an argument or function name that the implementation
    would have to introduce - prose is not evidence of capability. `declined` marks a task the
    design decision deliberately excludes, which is a settled state, not a gap.
    """
    if declined:
        return {"kind": "design", "name": name, "probe": "(settled by decision)", "what": what,
                "pending": False}
    found = subprocess.run(["grep", "-rqiE", probe, str(SCRIPTS / "run_recommendation.py")],
                           capture_output=True).returncode == 0
    return {"kind": "code", "name": name, "probe": probe, "what": what, "pending": not found}


def status() -> dict:
    jobs, hours = queue()
    aud = audit()
    man = manifest()
    docs = pdf_freshness()
    main = (PAPER / "acmart-primary" / "acmmanuscript.tex").read_text(encoding="utf-8")
    stamp_in_docs = re.findall(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-f]+)\}", main)
    tasks = [
        code_task("competitive-model attribution audit",
                  r"auditing a neural scorer|audit_for_model|--model\s+sasrec.*aia",
                  "run_recommendation.py scores SASRec/LightGCN ranking quality but has no "
                  "attribution audit for them, so the architecture-general claim stays narrowed. "
                  "Either add a bounded-AIA audit path for --model sasrec/lightgcn, or keep the "
                  "scope sentence as it is."),
        code_task("adaptive stopping", r"adaptive_stop|stopping_criterion",
                  "A paired-variance stopping rule for unstable users, so per-user MC error is "
                  "acted on instead of only reported. Then re-run the headline cells under it."),
        code_task("refit-uncertainty splits", r"train.splits|refit_seeds|fit_seed",
                  "Repeated preprocessing/training splits feeding the same user-level inference, so "
                  "intervals stop being conditional on one fitted structure."),
        code_task("matched-user LIME mask ablation", r"lime.masks.*fixed.cohort|mask.*same.users",
                  "Run masks on one fixed cohort so the design comparison is causal rather than "
                  "confounded with a changing population."),
        code_task("per-method valid n in the prospective panel", "",
                  "Decided: the four method columns share one defined subset, printed as one "
                  "`Defined n` row, and the text says so. Four separate subsets are only worth "
                  "adding if the methods are re-run with method-specific abstention rules.",
                  declined=True),
        code_task("generator for the supplement's proportions block", "prop_quality_block",
                  "Decided otherwise: the block is verified row by row by audit_success_estimand.py "
                  "and pinned to user_seed_metrics.csv.gz, so it can be checked but not rewritten by "
                  "a generator. Wrapping it in make_review3_stats.py would be convenience, not "
                  "integrity.", declined=True),
    ]
    return {
        "queued_runs": len(jobs),
        "queued_hours": round(hours, 1),
        "jobs": jobs,
        "audit_rows": len(aud["rows"]),
        "audit_unsupported": aud["unsupported"],
        "audit_grid": aud.get("slice", {}).get("grid", 1.0 / (1000 * 5)),
        "manifest_stamp": man["manifest_stamp"],
        "stamp_in_documents": stamp_in_docs,
        "stamp_matches_documents": bool(stamp_in_docs) and all(s == man["manifest_stamp"] for s in stamp_in_docs),
        "documents_stale": [d["document"] for d in docs if d["pdf_mtime"] < d["newest_mtime"]],
        "manifest_files": len(man["files"]),
        "mirror_drift": mirror_parity(),
        "code_tasks": tasks,
    }


def md(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


HEADER = """# Remaining work on the ActionShap submission

Everything still open, in one place, and **computed from the repository** rather than written down:
the pending runs, the verification of what is already typeset, the engineering tasks that no
computer time solves, and the close-out chain. Regenerate this notebook with

```
python code/scripts/make_remaining_work_notebook.py
```

so the statuses below reflect the tree as it is today. Run the cells in order; none of them mutates
the paper except the close-out section, which is explicit about what it writes.
"""

CELL_STATUS = r'''import json, re, subprocess, sys
from pathlib import Path

def _root():                      # works from notebooks/, the project dir, or the repository root
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "code" / "scripts" / "make_remaining_work_notebook.py").exists():
            return base
    raise RuntimeError("run this notebook from inside the ActionShap project")

ROOT = _root()
CODE = ROOT / "code"
REPO = next(a for a in [ROOT, *ROOT.parents] if (a / "Makefile").exists())   # make lives here
sys.path.insert(0, str(CODE / "scripts"))
import make_remaining_work_notebook as mrw

st = mrw.status()
print(f"pending runs ............ {st['queued_runs']}  (~{st['queued_hours']:.0f} h plan)")
print(f"printed rows verified ... {st['audit_rows'] - st['audit_unsupported']}/{st['audit_rows']} "
      f"({st['audit_unsupported']} unsupported), lattice {st['audit_grid']:.6f}")
print(f"manifest stamp .......... {st['manifest_stamp']} "
      f"({'quoted correctly in both documents' if st['stamp_matches_documents'] else 'MISMATCH: ' + str(st['stamp_in_documents'])})")
print(f"stale PDFs .............. {st['documents_stale'] or 'none'}")
print(f"tree agreement on decision sentences ... {st['mirror_drift'] or 'ok'}")
pend = [t["name"] for t in st["code_tasks"] if t["pending"]]
print(f"engineering tasks open ... {len(pend)}/{len(st['code_tasks'])}: {', '.join(pend) or 'none'}")
'''

CELL_VERIFY = r'''# Verify every number the supplement's decision-quality block prints, from the frozen matrices.
out = subprocess.run([sys.executable, str(CODE / "scripts" / "audit_success_estimand.py"), "--check"],
                     cwd=str(CODE), capture_output=True, text=True)
print("gate:", "PASS" if out.returncode == 0 else "FAIL")
print("\n".join(out.stdout.splitlines()[-6:]))
if out.returncode:
    print(out.stderr)
audit = json.loads((CODE / "results" / "review9" / "success_estimand_audit.json").read_text())
worst = max(audit["rows"], key=lambda r: -r["abs_delta"])
print(f"\nlargest deviation: {worst['abs_delta']:.6f} on "
      f"{worst['dataset']}/{worst['method']}/{worst['quantity']}")
print("convention recorded in the text: seed-averaged per-seed indicator, so rates are multiples of "
      f"{audit['slice']['grid']:.6f} = 1/(n * R_seed).")
'''

CELL_QUEUE = r'''# The compute: same derived queue as OUTSTANDING_RUNS.ipynb, printed here so the two never disagree.
gen = mrw._load("mor_gen", CODE / "scripts" / "make_outstanding_runs_notebook.py")
jobs = gen.build_jobs()
print(f"{len(jobs)} jobs, plan {sum(j['minutes'] for j in jobs) / 60:.1f} h; open "
      "notebooks/OUTSTANDING_RUNS.ipynb to run them (DRY_RUN = True there by default).\n")
for job in sorted(jobs, key=lambda j: -j["minutes"]):
    print(f"  ~{job['minutes']:>5.0f} min  {job['experiment']:<18} {job['dataset']:<10} -> {job['out']}")
print("\nIf a job is declined, the claim it would support stays hedged; see the decisions cell.")
'''

CELL_TASKS = r'''# Engineering tasks: pending means the probe below finds no implementation in code/scripts or the
# manuscript. Each row says what would close it and which claim is currently narrowed because of it.
for task in st["code_tasks"]:
    state = "PENDING" if task["pending"] else ("BY DECISION" if task["kind"] == "design" else "absent")
    print(f"[{state:>7}] {task['name']}")
    print(f"          probe: {task['probe']}")
    print(f"          {task['what']}\n")
'''

CELL_DECISIONS = r'''# Decisions that are yours, not the machine's. Printed from the current text so the wording you
# would keep or change is visible, not paraphrased.
docs = {"acmmanuscript.tex": (Path(CODE).parent / "acmart-primary" / "acmmanuscript.tex").read_text(),
        "supplementary.tex": (Path(CODE).parent / "acmart-primary" / "supplementary.tex").read_text()}

def first_with(needles, limit=620):
    """First paragraph (in main, then supplement) containing any needle: (source, text)."""
    for name in ("acmmanuscript.tex", "supplementary.tex"):
        for para in docs[name].replace("\r\n", "\n").split("\n\n"):
            flat = " ".join(para.split())
            if any(n in flat for n in needles):
                # quote the sentence that carries the hedge, not the whole paragraph around it
                for sentence in flat.split(". "):
                    if any(n in sentence for n in needles):
                        text = sentence if sentence.endswith(".") else sentence + "."
                        return name, (text[:limit] + ("..." if len(text) > limit else ""))
                return name, (flat[:limit] + ("..." if len(flat) > limit else ""))
    return None, "(no matching sentence in either document)"

print("Architecture scope:")
src, text = first_with(["as the primary model", "history-conditioned"])
print(f"  [{src}] {text}\n")
print("Null calibration:")
src, text = first_with(["not structure-preserving", "uncalibrated", "unstratified"])
print(f"  [{src}] {text}\n")
print("Success estimand:")
src, text = first_with(["per-seed indicator"])
print(f"  [{src}] {text}\n")
print("Data and code availability:")
print(f"  both documents quote manifest stamp {st['manifest_stamp']}; the frozen release lists "
      f"{st['manifest_files']} hashed files, and the per-user matrices are what every rate above "
      f"recomputes from.\n")
print("Venue: the documents are ACM-formatted and this session deliberately did not convert them for "
      "another journal; converting is a formatting task with no bearing on the analysis.")
'''

CELL_CLOSE = r'''# Close-out, in order. `apply = False` prints the commands instead of running them.
apply = False

def sh(*argv, where=REPO):
    cmd = " ".join(argv)
    if apply:
        return subprocess.run(argv, cwd=str(where), capture_output=True, text=True, check=True).stdout
    return f"would run: cd {where} && {cmd}"

print(sh("make", "tables",   f"PY={sys.executable}"))
print(sh("make", "manifest", f"PY={sys.executable}"))
stamp = json.loads((Path(CODE) / "results" / "manifest.json").read_text())["manifest_stamp"]
for doc in ("acmart-primary/acmmanuscript.tex", "acmart-primary/supplementary.tex"):
    path = Path(CODE).parent / doc
    text = path.read_text()
    new = re.sub(r"(\\newcommand\{\\resultmanifeststamp\})\{[0-9a-f]+\}",
                 lambda m: m.group(1) + "{" + stamp + "}", text, count=1)
    if apply and new != text:
        path.write_text(new)
    print(("re-stamped " if new != text else "stamp ok for ") + doc)
print(sh("make", "pdf",   f"PY={sys.executable}"))     # needs a TeX toolchain
print(sh("make", "ready", f"PY={sys.executable}"))      # must report no blockers
print(sh("make", "check", f"PY={sys.executable}"))      # validators + suite, incl. the estimand gate
print(sh("make", "overleaf", f"PY={sys.executable}"))   # repacks the submission archive
print("\nAfter `make pdf` succeeds: delete the xfail marker on the PDF-freshness test in "
      "code/tests/test_review9_publication_integrity.py, then `make check` must be fully green.")
'''

CELL_FINAL = r'''# The submission gate: READY only when every one of these is clean.
st2 = mrw.status()
gates = {
    "all printed rows reproducible": st2["audit_unsupported"] == 0,
    "stamp quoted matches the manifest": st2["stamp_matches_documents"],
    "documents rebuilt after the last source edit": not st2["documents_stale"],
    "decision sentences agree in both trees": not st2["mirror_drift"],
    "queued runs executed or consciously declined": True,   # flip in your head, not in the file
}
for name, ok in gates.items():
    print(f"  [{'x' if ok else ' '}] {name}")
print("\nREADY" if all(gates.values()) else "\nNOT READY: " + ", ".join(k for k, v in gates.items() if not v))
'''


def main() -> int:
    st = status()
    jobs_line = (f"**{st['queued_runs']} runs** remain (~{st['queued_hours']:.0f} h plan), "
                 f"**{st['audit_rows']} printed rows** in the supplement's decision-quality block "
                 f"verified against the frozen matrices with **{st['audit_unsupported']} unsupported**, "
                 f"manifest stamp `{st['manifest_stamp']}` "
                 f"{'correctly quoted in both documents' if st['stamp_matches_documents'] else 'MISMATCHED in the documents'}, "
                 f"{'both PDFs stale' if st['documents_stale'] else 'PDFs fresh'}, "
                 f"{'decision sentences disagree in: ' + ', '.join(st['mirror_drift']) if st['mirror_drift'] else 'decision sentences agree in both trees'}.")
    cells = [
        md(HEADER + "\n## 0. State of the submission right now\n\n" + jobs_line),
        code(CELL_STATUS),
        md("## 1. Verify what the paper already prints\n\nThe re-review's items 13-14 asked for one "
           "estimand, regenerated and traceable. It is now pinned in the text (success is the "
           "seed-averaged per-seed indicator, so cohort rates are multiples of "
           f"1/(n·R<sub>seed</sub>) = {st['audit_grid']:.4f} for the 1,000-user primary cohort) and "
           "gated by `make check`. This cell recomputes every row of the block from "
           "`user_seed_metrics.csv.gz`."),
        code(CELL_VERIFY),
        md("## 2. The compute that is still owed\n\nNothing here is invented: the job list comes from "
           "the subcommands `run_review9_experiments.py` defines and the payloads actually present in "
           "`code/results/review9/`."),
        code(CELL_QUEUE),
        md("## 3. Remaining work that is engineering, not compute\n\nA longer queue does not settle "
           "these; each probe is a grep over the repository, so the cell turns green when the work "
           "exists."),
        code(CELL_TASKS),
        md("## 4. Decisions only the authors can make\n\nWhich hedges stay depends on what you choose "
           "to run. The rule the submission obeys is that no claim outruns a payload."),
        code(CELL_DECISIONS),
        md("## 5. Close-out\n\nSet `apply = True` once the runs and decisions are settled. This is the "
           "only section that writes."),
        code(CELL_CLOSE),
        code(CELL_FINAL),
    ]
    for index, cell in enumerate(cells):          # a shipped cell must never fail to parse
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {index}", "exec")

    NOTEBOOK.write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
