"""The hand-off queue must match reality: same jobs, real flags, safe defaults.

This notebook is what the authors run on their own machine, so two failures matter: a queue entry
whose command does not exist (the run dies an hour in), and a queue that has drifted from the
payloads actually present (re-run work or, worse, skip it).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
SCRIPTS = CODE / "scripts"
NOTEBOOK = CODE.parent / "notebooks" / "OUTSTANDING_RUNS.ipynb"


def _load_generator():
    spec = importlib.util.spec_from_file_location("mor", SCRIPTS / "make_outstanding_runs_notebook.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mor"] = module
    spec.loader.exec_module(module)
    return module


def test_the_queue_matches_the_missing_payloads():
    gen = _load_generator()
    queued = {(j["experiment"], j["dataset"]) for j in gen.build_jobs()}
    recomputed = set()
    for experiment in gen.PLAN:
        for dataset in gen.DATASETS:
            path = CODE / "results" / "review9" / f"{experiment.replace('-', '_')}_{dataset}.json"
            if not path.exists():
                recomputed.add((experiment, dataset))
                continue
            payload = json.loads(path.read_text())
            keys = gen.REQUIRED_KEYS.get(experiment, ())
            if any(k not in payload or (isinstance(payload[k], (list, dict)) and not payload[k])
                   for k in keys):
                recomputed.add((experiment, dataset))
    assert queued == recomputed, (sorted(queued - recomputed), sorted(recomputed - queued))
    # an empty results/review9 must queue all 21 jobs, so a fresh checkout cannot under-report
    assert len(gen.PLAN) * len(gen.DATASETS) == 21


def test_every_queued_command_is_a_real_subcommand_with_real_flags():
    gen = _load_generator()
    runner = (SCRIPTS / "run_review9_experiments.py").read_text(encoding="utf-8")
    subcommands = set(re.findall(r'"([a-z-]+)": cmd_', runner))
    declared_flags = set(re.findall(r'add_argument\("--([a-z-]+)"', runner))
    assert subcommands, "the runner uses subparsers; the parser changed shape"
    for job in gen.build_jobs():
        argv = job["argv"]
        assert argv[0] in subcommands, f"{argv[0]} is not a subcommand of the runner"
        for token in argv:
            if token.startswith("--"):
                assert token[2:] in declared_flags, f"{token} is not accepted by {argv[0]}"
        assert Path(job["out"]).name == f"{job['experiment'].replace('-', '_')}_{job['dataset']}.json"
        assert "--users" in argv and "--dataset" in argv


def _embedded_jobs(config: str):
    """The job list literal passed to json.loads in the config cell, parsed from the AST."""
    tree = ast.parse(config)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "JOBS" for t in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)):
            return json.loads(node.value.args[0].value)
    return None


def test_the_notebook_exists_and_defaults_to_a_dry_run():
    gen = _load_generator()
    assert NOTEBOOK.exists(), "run `python code/scripts/make_outstanding_runs_notebook.py`"
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    code_cells = [s for s, c in zip(sources, notebook["cells"]) if c["cell_type"] == "code"]
    for src in code_cells:
        ast.parse(src)                      # every emitted cell must at least be valid Python
    config = next(s for s in code_cells if "DRY_RUN" in s)
    assert "DRY_RUN   = True" in config, "a 100-hour queue must not execute on first run"
    assert "assert Path(\"scripts/run_review9_experiments.py\").exists()" in config
    jobs = _embedded_jobs(config)
    assert jobs is not None, "the config cell no longer embeds its job list"
    # The notebook gates on payload presence at run time, so as runs finish the derived queue shrinks
    # while the committed plan stays the record of what was queued. The invariant that protects the
    # paper is coverage: no outstanding job may be missing from the notebook.
    outstanding = {j["out"] for j in gen.build_jobs()}
    assert outstanding <= {j["out"] for j in jobs}, sorted(outstanding - {j["out"] for j in jobs})
    assert jobs == gen.build_jobs(), "the notebook's job list differs from the live queue"
    markdown = "\n".join(s for s, c in zip(sources, notebook["cells"]) if c["cell_type"] == "markdown")
    # the notebook must be honest about what it does not cover
    for phrase in ("no CPU on this machine fixes", "not to run something"):
        assert phrase in markdown
    # and it must not quote review-round bookkeeping as if the paper were a workshop response
    assert not re.search(r"[Rr]eview-\d|\(Issue \d+\)", markdown)
