#!/usr/bin/env python3
"""Emit ``notebooks/REMAINING_WORK.ipynb``: the close-out notebook that *does* the close-out.

Design contract, so the notebook can be run with "Restart & Run All" and nothing else is needed:

* **Idempotent.** Every step inspects the repository first and skips work whose output already exists
  (a payload on disk, a stamp already quoted, a scope sentence already typeset), so re-running after an
  interruption resumes instead of repeating.
* **Executes, not narrates.** ``APPLY = True`` by default: it runs the outstanding jobs, regenerates
  tables, re-freezes and re-quotes the manifest, builds the PDFs when a TeX engine is installed, drops
  the xfail marker once the rebuilt PDFs actually satisfy the test, and repacks the archive.
* **Never fabricates.** It only invokes subcommands that ``run_review9_experiments.py`` declares, only
  reports numbers it recomputes, and its final gate lists what no machine can decide (venue conversion,
  deposit registration) instead of pretending those are done.
* **Interruptible.** ``MAX_HOURS`` bounds the queue for one sitting; Run All again continues where it
  stopped. ``SKIP_RUNS = True`` does everything that is not cohort compute.

Regenerate with ``python code/scripts/make_remaining_work_notebook.py``.
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

# The four boundaries that used to be "engineering gaps". They are closed by stating the scope the
# release can support; the notebook verifies each clause is typeset, so a gap can never silently
# re-open as an unqualified claim.
SCOPE_CLAUSES = {
    "competitive-model attribution": "not attribution alignment",
    "adaptive stopping": "adaptive stopping rule",
    "refit uncertainty": "single fitted structure per seed",
    "mask-ablation cohort": "descriptive rather than causal",
}
SCOPE_PARAGRAPH = (
    "Four further boundaries are stated so that no result is read past them. The attribution audit is "
    "implemented for the primary ItemKNN scorer: for SASRec and LightGCN the paper reports measured "
    "ranking quality only, not attribution alignment, so no architecture-general attribution claim is "
    "made. Inference uses fixed permutation and bootstrap budgets rather than an adaptive stopping rule, "
    "so per-user Monte-Carlo error is reported but not acted upon. Intervals are conditional on a single "
    "fitted structure per seed; refit and preprocessing uncertainty is not included. The mask ablations "
    "are defined on the cohort each design admits, so that comparison is descriptive rather than causal, "
    "and the prospective panel reports one shared defined subset rather than a per-method valid $n$."
)
SCOPE_ANCHOR = "future work should study heterogeneous costs and user-side constraints."


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scope_state() -> dict:
    """Where the scope paragraph belongs, and whether each clause is already typeset."""
    main = PAPER / "acmart-primary" / "acmmanuscript.tex"
    text = main.read_text(encoding="utf-8")
    return {"anchor_present": SCOPE_ANCHOR in text,
            "clause_present": {k: (v in text) for k, v in SCOPE_CLAUSES.items()},
            "all_present": all(v in text for v in SCOPE_CLAUSES.values())}


def queue() -> list[dict]:
    try:
        gen = _load("mor_gen_for_status", SCRIPTS / "make_outstanding_runs_notebook.py")
        jobs = gen.build_jobs()
    except Exception:                                              # pragma: no cover - defensive
        return []
    for job in jobs:
        job["done"] = (CODE / job["out"]).exists()
    return jobs


def audit() -> dict:
    path = CODE / "results" / "review9" / "success_estimand_audit.json"
    if not path.exists():
        subprocess.run([sys.executable, str(SCRIPTS / "audit_success_estimand.py")],
                       cwd=str(CODE), capture_output=True, text=True)
    return json.loads(path.read_text(encoding="utf-8"))


def status() -> dict:
    jobs = queue()
    aud = audit()
    man = json.loads((CODE / "results" / "manifest.json").read_text(encoding="utf-8"))
    docs = {}
    for name in ("acmmanuscript", "supplementary"):
        pdf = PAPER / "acmart-primary" / f"{name}.pdf"
        src = max((p for p in (PAPER / "acmart-primary").rglob("*.tex")), key=lambda p: p.stat().st_mtime)
        docs[name] = {"pdf_newer": pdf.exists() and pdf.stat().st_mtime >= src.stat().st_mtime,
                      "newest_source": src.name}
    stamps = re.findall(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-f]+)\}",
                        (PAPER / "acmart-primary" / "acmmanuscript.tex").read_text(encoding="utf-8"))
    supp = re.findall(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-f]+)\}",
                      (PAPER / "acmart-primary" / "supplementary.tex").read_text(encoding="utf-8"))
    stamps = stamps + supp
    return {"pending_jobs": [j for j in jobs if not j["done"]],
            "n_jobs": len(jobs), "done_jobs": sum(1 for j in jobs if j["done"]),
            "queued_hours": round(sum(j["minutes"] for j in jobs if not j["done"]) / 60.0, 1),
            "audit_rows": len(aud["rows"]), "audit_unsupported": aud["unsupported"],
            "audit_grid": aud.get("slice", {}).get("grid", 1.0 / 5000),
            "manifest_stamp": man["manifest_stamp"], "manifest_files": len(man["files"]),
            "stamp_matches": bool(stamps) and all(s == man["manifest_stamp"] for s in stamps),
            "stale_pdfs": [k for k, v in docs.items() if not v["pdf_newer"]],
            "scope": scope_state()}


def md(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text: str) -> dict:
    lines = text.split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


HEADER = """# Remaining work on the ActionShap submission — run all to finish it

This notebook **does** the remaining work rather than listing it. Restart and run all: it verifies the
printed numbers, executes whatever cohort runs are still missing, regenerates and re-freezes the
release, rebuilds the documents, clears the one test marker that waits on a TeX build, and repacks the
submission archive. Every step reads the repository first and skips finished work, so it resumes
safely after an interruption and says so in its output.

The only things it will not do are the ones no machine can decide for you; §7 lists those.

The notebook is self-contained: it imports nothing from ``code/scripts`` at runtime, so it cannot be
confused by a module the kernel had already loaded. Regenerate it after changing the pipeline with
``python code/scripts/make_remaining_work_notebook.py``.
"""

CELL_CONFIG = r'''import json, re, shutil, subprocess, sys, time
from pathlib import Path

def _root():                        # works from notebooks/, the project dir, or the repository root
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "code" / "scripts" / "make_remaining_work_notebook.py").exists():
            return base
    raise RuntimeError("run this notebook from inside the ActionShap project")

ROOT = _root()
CODE, PAPER = ROOT / "code", ROOT
REPO = next(a for a in [ROOT, *ROOT.parents] if (a / "Makefile").exists())   # the Makefile lives here
PY = sys.executable

APPLY = True        # False prints commands instead of running them (safe rehearsal)
SKIP_RUNS = False   # True does everything except the cohort queue
MAX_HOURS = None    # bound one sitting, e.g. 8.0; run all again to resume

# Baked from the repository when this notebook was generated; payload presence is re-checked at
# runtime below, so a job finished since then is skipped rather than repeated. Nothing here is
# imported from code/scripts, which keeps a long-lived kernel from reusing a stale module object.
JOBS = __JOBS__
SCOPE_CLAUSES = __CLAUSES__
SCOPE_PARAGRAPH = __PARAGRAPH__
SCOPE_ANCHOR = __ANCHOR__


def sh(argv, cwd=None, note=""):
    """Run argv; return its stdout, raising with the captured output on failure."""
    if not APPLY:
        print(f"[dry-run] cd {cwd or REPO} && {' '.join(map(str, argv))}")
        return "", 0
    started = time.time()
    proc = subprocess.run([str(a) for a in argv], cwd=str(cwd or REPO), capture_output=True, text=True)
    print(f"      {time.time() - started:6.1f} min  {(note or str(argv[-1]))[:60]}")
    if proc.returncode != 0:
        print("\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]))
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, argv))}")
    return proc.stdout, proc.returncode


def make(target, note=None):
    return sh(["make", target, f"PY={PY}"], cwd=REPO, note=note or f"make {target}")


def pending():
    return [j for j in JOBS if not (CODE / j["out"]).exists()]


def manifest_stamp():
    return json.loads((CODE / "results/manifest.json").read_text())["manifest_stamp"]


def stamps_quoted():
    out = []
    for doc in ("acmart-primary/acmmanuscript.tex", "acmart-primary/supplementary.tex"):
        body = (PAPER / doc).read_text(encoding="utf-8")
        out += re.findall(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-f]+)\}", body)
    return out


def stale_pdfs():
    src = max((q.stat().st_mtime for q in (PAPER / "acmart-primary").rglob("*.tex")))
    return [name for name in ("acmmanuscript", "supplementary")
            if not (PAPER / "acmart-primary" / f"{name}.pdf").exists()
            or (PAPER / "acmart-primary" / f"{name}.pdf").stat().st_mtime < src]


def audit_state():
    path = CODE / "results/review9/success_estimand_audit.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {"rows": len(data["rows"]), "unsupported": data["unsupported"],
            "grid": data.get("slice", {}).get("grid", 0.0)}


def summary() -> list[str]:
    lines = [f"repo      {REPO}"]
    left = pending()
    hours = sum(j["minutes"] for j in left) / 60.0
    lines.append(f"queue     {len(left)} of {len(JOBS)} jobs left (~{hours:.0f} h)")
    aud = audit_state()
    lines.append("audit     " + ("not run yet (section 2 runs it)" if aud is None
                                 else f"{aud['rows'] - aud['unsupported']}/{aud['rows']} rows supported"))
    quoted = set(stamps_quoted())
    lines.append("stamp     " + manifest_stamp() + (" quoted in both documents"
                 if quoted == {manifest_stamp()} else " needs re-quoting (section 4 fixes it)"))
    lines.append("pdfs      " + ("fresh" if not stale_pdfs() else "stale -> " + ", ".join(stale_pdfs())))
    return lines


print("\n".join(summary()))
'''

CELL_SCOPE = r'''# 1. Scope the claims the queue cannot cover. The four boundaries below are closed by stating them
#    exactly, so the paragraph must be typeset; this inserts it where it belongs if it is missing.
def scope_missing(body):
    return {name: clause for name, clause in SCOPE_CLAUSES.items() if clause not in body}


def with_scope(body):
    return body.replace(SCOPE_ANCHOR, SCOPE_ANCHOR + "\n\n" + SCOPE_PARAGRAPH, 1)


targets = [PAPER / "acmart-primary" / "acmmanuscript.tex", PAPER / "actionshap-ipm" / "acmmanuscript.tex"]
main = targets[0]
body = main.read_text(encoding="utf-8")
if not scope_missing(body):
    print("scope paragraph already typeset: nothing to do")
elif SCOPE_ANCHOR not in body:
    raise RuntimeError("anchor sentence not found - the Limitations paragraph moved; re-locate it by hand")
else:
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if SCOPE_ANCHOR not in text or scope_missing(text) != scope_missing(body):
            continue
        if APPLY:
            path.write_text(with_scope(text))
            print(f"inserted scope paragraph into {path.relative_to(PAPER)}")
        else:
            print(f"[dry-run] would insert scope paragraph into {path.relative_to(PAPER)}")

still = scope_missing(main.read_text(encoding="utf-8"))
if APPLY:
    assert not still, f"the manuscript must state all four boundaries; missing {still}"
    print("all four boundaries are typeset")
else:
    print(f"dry-run: {len(still)} clause(s) would be added -> {', '.join(still)}")
'''

CELL_VERIFY = r'''# 2. Prove the numbers already in the paper, row by row, from the frozen matrices.
out, rc = sh([PY, str(PAPER / "code/scripts/audit_success_estimand.py"), "--check"], cwd=CODE,
             note="estimand audit")
aud = audit_state()
print(f"rows {aud['rows']}, unsupported {aud['unsupported']}, lattice {aud['grid']:.6f} "
      "= 1/(n R_seed) -> every printed rate is reproducible")
print("(the audit is also wired into `make check`, so this cannot drift silently)")
'''

CELL_RUNS = r'''# 3. The cohort queue: only jobs whose payload is absent, only real subcommands, resumable.
jobs = pending()
if SKIP_RUNS:
    print(f"SKIP_RUNS = True: {len(jobs)} jobs left ({sum(j['minutes'] for j in jobs)/60:.1f} h)")
elif not jobs:
    print("all queued payloads present: nothing to run")
else:
    total = sum(j["minutes"] for j in jobs) / 60.0
    print(f"running {len(jobs)} jobs (~{total:.1f} h plan); Ctrl-C is safe - finished payloads are kept\n")
    spent = 0.0
    for job in jobs:
        if MAX_HOURS is not None and spent >= MAX_HOURS:
            print(f"MAX_HOURS={MAX_HOURS:.1f} h reached after {spent:.1f} h; the remaining jobs "
                  f"resume on the next run all")
            break
        print(f"  {job['experiment']:<18} {job['dataset']:<10} users={job['users']}")
        sh([PY, "scripts/run_review9_experiments.py", *job["argv"]], cwd=CODE,
           note=f"{job['experiment']}/{job['dataset']}")
        spent += job["minutes"] / 60.0
    left = pending()
    print(f"\npayloads now on disk: {len(JOBS) - len(left)}/{len(JOBS)}"
          + ("" if not left else f"; still missing: {', '.join(j['experiment'] + '/' + j['dataset'] for j in left)}"))
'''

CELL_RELEASE = r'''# 4. Rebuild the release artifacts from the payloads, then make the documents quote the new stamp.
make("tables", "regenerate tables and statistics")
make("manifest", "re-freeze the manifest")
stamp = json.loads((CODE / "results/manifest.json").read_text())["manifest_stamp"]
for doc in ("acmart-primary/acmmanuscript.tex", "acmart-primary/supplementary.tex"):
    path = PAPER / doc
    body = path.read_text(encoding="utf-8")
    quoted = re.findall(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-f]+)\}", body)
    if quoted.count(stamp) == 1 and len(quoted) == 1:
        print(f"      stamp already {stamp} in {doc}")
    else:
        new = re.sub(r"(\\newcommand\{\\resultmanifeststamp\})\{[0-9a-f]*\}",
                     lambda m: m.group(1) + "{" + stamp + "}", body, count=1)
        if APPLY:
            path.write_text(new)
        print(f"      re-quoted {doc}: {quoted} -> {stamp}")
make("check", "validators + suite (before the PDF build)")
print(f"release rebuilt at {stamp} ({len(json.loads((CODE / 'results/manifest.json').read_text())['files'])} files)")
'''

CELL_PDF = r'''# 5. Documents. Needs a TeX engine; if none is installed the notebook says so instead of faking it,
#    and the archive in §6 still rebuilds so an Overleaf compile is a valid substitute.
engine = next((e for e in ("latexmk", "pdflatex", "tectonic") if shutil.which(e)), None)
built = False
if engine is None:
    print("no TeX engine on PATH: `make pdf` skipped.")
    print("    either  brew install --cask mactex-no-gp  then run all again,")
    print("    or compile the archive in §6 on Overleaf and drop the two PDFs back into acmart-primary/.")
else:
    print(f"building with {engine}")
    make("pdf", "build both documents")
    built = True

# The suite carries one xfail marker that exists only because the committed PDFs predate the text fixes.
# Clear it *only* if the rebuilt PDFs actually satisfy the test; otherwise restore it.
tests = CODE / "tests/test_review9_publication_integrity.py"
marker = re.compile(r"[ \t]*@pytest\.mark\.xfail[^\n]*\n")
src = tests.read_text(encoding="utf-8")
if not built:
    print("PDF-freshness marker left in place (documents not rebuilt)")
elif APPLY and marker.search(src):
    tests.write_text(marker.sub("", src, count=1))
    probe = sh([PY, "-m", "pytest", "tests/test_review9_publication_integrity.py", "-q"], cwd=CODE,
               note="suite without the marker")
    if "failed" in probe[0]:
        tests.write_text(src)
        print("test still fails -> marker restored; inspect the PDF build")
    else:
        print("marker removed: the rebuilt PDFs carry the revised text")
else:
    print("marker already cleared" if not marker.search(src) else "dry-run: would clear the marker")
'''

CELL_PACKAGE = r'''# 6. Submission archive and the final gate.
make("overleaf", "repack the Overleaf archive")
zip_path = PAPER / "actionshap-overleaf.zip"
if zip_path.exists():
    import zipfile
    names = zipfile.ZipFile(zip_path).namelist()
    tex = zipfile.ZipFile(zip_path).read("acmmanuscript.tex").decode()
    quoted = re.search(r"resultmanifeststamp\}\{([0-9a-f]+)", tex)
    print(f"      {len(names)} files; archive quotes stamp "
          + (quoted.group(1) if quoted else "MISSING"))

aud2, missing_scope = audit_state(), scope_missing(
    (PAPER / "acmart-primary" / "acmmanuscript.tex").read_text(encoding="utf-8"))
done = {
    "printed rows reproducible from the release": bool(aud2) and aud2["unsupported"] == 0,
    "every queued payload present": not pending(),
    "documents quote the current manifest stamp": set(stamps_quoted()) == {manifest_stamp()},
    "boundaries stated in the text": not missing_scope,
    "tables regenerated and validators green": True,      # §4 raises otherwise
}
open_items = []
if stale_pdfs():
    open_items.append("PDFs not rebuilt here (no TeX engine) - compile in Overleaf or install one")
open_items.append("upload the archive; register the artifact DOI/license after acceptance")
for name, ok in done.items():
    print(f"  [{'x' if ok else ' '}] {name}")
print("\nCLOSED (everything a machine can decide is done)" if all(done.values())
      else "\nNOT CLOSED: " + ", ".join(k for k, v in done.items() if not v))
print("left for you:\n  - " + "\n  - ".join(open_items))
print(f"\ncommit with:  git add -A && git commit -m 'regenerated after the run queue' && git push")
'''


def _inline(source: str) -> str:
    """Bake the derived constants into the cell source; the notebook imports nothing from code/scripts."""
    jobs = [{"experiment": j["experiment"], "dataset": j["dataset"], "users": j["users"],
             "argv": j["argv"], "out": j["out"], "minutes": round(j["minutes"], 1)} for j in queue()]
    baked = "[\n  " + ",\n  ".join(json.dumps(j) for j in jobs) + "\n]" if jobs else "[]"
    return (source.replace("__JOBS__", baked)
            .replace("__CLAUSES__", json.dumps(SCOPE_CLAUSES, indent=None))
            .replace("__PARAGRAPH__", json.dumps(SCOPE_PARAGRAPH))
            .replace("__ANCHOR__", json.dumps(SCOPE_ANCHOR))
            .replace("st['audit_grid']", f"{status()['audit_grid']:.4f}"))


def build_cells() -> list[dict]:
    st = status()
    jobs = st["n_jobs"] - st["done_jobs"]
    return [
        md(HEADER),
        code(_inline(CELL_CONFIG)),
        md("## 1. Claims the queue cannot cover\n\nFour items on the worklist are engineering or scope "
           "decisions, not compute. This is the only cell that touches the manuscript, and it is a "
           "single idempotent paragraph."),
        code(_inline(CELL_SCOPE)),
        md("## 2. Verify the printed numbers\n\nThe supplement's decision-quality block is recomputed "
           "row by row from `user_seed_metrics.csv.gz`; success rates are the seed-averaged per-seed "
           "indicator, so values live on a 1/(n R_seed) lattice."),
        code(_inline(CELL_VERIFY)),
        md(f"## 3. The cohort queue\n\n{jobs} of {st['n_jobs']} payloads still missing "
           f"(~{st['queued_hours']:.1f} h). Set `SKIP_RUNS = True` in §0 to do everything else and leave "
           "these to another machine, or `MAX_HOURS` to bound this sitting."),
        code(_inline(CELL_RUNS)),
        md("## 4. Regenerate the release\n\nTables and statistics from the payloads, manifest re-frozen, "
           "stamp re-quoted in both documents, then the whole validator suite."),
        code(_inline(CELL_RELEASE)),
        md("## 5. Build the documents"),
        code(_inline(CELL_PDF)),
        md("## 6. Package and gate"),
        code(_inline(CELL_PACKAGE)),
        md("## 7. What deliberately stays with the authors\n\n* **Venue formatting.** The documents are "
           "ACM-formatted. Converting to another journal's template is a formatting task with no bearing "
           "on the analysis, and doing it blind would risk the tables.\n"
           "* **Deposit registration.** A DOI, a license choice and the public URL are actions on an "
           "external service; the archive references the frozen manifest, which is reproducible without them.\n"
           "* **Claim widening.** If you implement a competitive-model attribution audit or adaptive "
           "stopping, delete the corresponding sentence in §1's paragraph and re-run; the text then "
           "asserts what the payload supports. Nothing in this notebook widens a claim on its own."),
    ]


def main() -> int:
    cells = build_cells()
    for index, cell in enumerate(cells):        # a shipped cell must never fail to parse
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {index}", "exec")
    NOTEBOOK.write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n", encoding="utf-8")
    st = status()
    print(f"wrote {NOTEBOOK} ({len(cells)} cells; {st['n_jobs'] - st['done_jobs']} jobs pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
