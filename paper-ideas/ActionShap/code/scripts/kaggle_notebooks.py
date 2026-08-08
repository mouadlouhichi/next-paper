#!/usr/bin/env python3
"""Push, run, update, and pull the ActionShap notebook on Kaggle.

This script drives the official `kaggle` CLI (https://github.com/Kaggle/kaggle-api)
end to end for the canonical ActionShap Run-All notebook:

    prepare   Assemble the Kaggle bundle under code/kaggle/bundle/ (offline).
    push      prepare + upload the bundle and trigger a cloud run.
              Re-pushing the same slug UPDATES the kernel and starts a new run.
    status    Poll the run state until complete / error / canceled.
    output    Download the run's output files.
    pull      Download the executed notebook (with outputs embedded).
    list      List your kernels on Kaggle.
    verify    Validate everything locally (no network required).

Prerequisites
-------------
* `kaggle` CLI installed  (pip install kaggle)
* ~/.kaggle/kaggle.json with username + API key (chmod 600)
* Network access to api.kaggle.com (note: the Arena sandbox currently blocks
  Kaggle egress at the TLS layer; run this from a machine that can reach it).

Dataset strategy on Kaggle
--------------------------
By default the notebook downloads MovieLens-1M and Amazon Digital Music from
the GroupLens/UCSD mirrors using Kaggle's open internet egress (enable_internet
is always True). To avoid the download entirely, mount Kaggle-hosted mirrors
and point the pipeline at them:

    --dataset-slug <owner>/<dataset>        adds a dataset source to the kernel
    --movielens-input /kaggle/input/<slug>/ml-1m.zip
    --amazon-input    /kaggle/input/<slug>/Digital_Music_5.json.gz
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # paper-ideas/ActionShap
CODE_ROOT = REPO_ROOT / "code"
KAGGLE_DIR = CODE_ROOT / "kaggle"
BUNDLE_DIR = KAGGLE_DIR / "bundle"
OUTPUT_DIR = KAGGLE_DIR / "output"
PULLED_DIR = KAGGLE_DIR / "pulled"
SOURCE_NOTEBOOK = CODE_ROOT / "ActionShap_All.ipynb"
KAGGLE_NOTEBOOK = "ActionShap_All_kaggle.ipynb"
DEFAULT_SLUG = "actionshap-rev4"
# Keep the title slug-compatible with DEFAULT_SLUG: Kaggle derives a new
# kernel's slug from its title, and a mismatch produces a CLI warning and
# surprising update behavior. "ActionShap Rev4" -> "actionshap-rev4".
DEFAULT_TITLE = "ActionShap Rev4"
BOOTSTRAP_MARKER = "# ==== KAGGLE BOOTSTRAP"

NETWORK_HINT = (
    "\n[!] The Kaggle API did not respond. If the error mentions SSL/TLS or a "
    "closed connection, the machine you are on cannot reach api.kaggle.com "
    "(the Arena sandbox blocks Kaggle egress). Run these commands from your "
    "own machine, where ~/.kaggle/kaggle.json exists and the API is reachable."
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def find_kaggle_json() -> Path | None:
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir) / "kaggle.json")
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_username() -> str:
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user
    config = find_kaggle_json()
    if config is not None:
        try:
            return str(json.loads(config.read_text())["username"])
        except (KeyError, json.JSONDecodeError):
            pass
    raise RuntimeError(
        "Cannot determine your Kaggle username. Set KAGGLE_USERNAME or "
        "~/.kaggle/kaggle.json (or pass --username)."
    )


def find_kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "kaggle",
        Path.home() / "kaggle-venv" / "bin" / "kaggle",
        Path("/usr/local/bin/kaggle"),
    ):
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "The `kaggle` CLI is not installed. Install it with:  pip install kaggle"
    )


def run_cli(command: list[str], dry_run: bool, capture: bool = True):
    print(f"$ {shlex.join(command)}")
    if dry_run:
        return 0, "", ""
    try:
        proc = subprocess.run(command, capture_output=capture, text=True)
    except FileNotFoundError:
        print(f"[!] executable not found: {command[0]}")
        return 127, "", ""
    if proc.returncode != 0 and capture and "SSL" in proc.stderr:
        print(NETWORK_HINT)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
    )


# ---------------------------------------------------------------------------
# Bundle preparation (offline)
# ---------------------------------------------------------------------------

def bootstrap_cells(slug: str) -> list[dict]:
    markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Kaggle cloud run\n",
            "This notebook was prepared by `scripts/kaggle_notebooks.py`. On Kaggle, "
            "the pushed bundle arrives read-only under `/kaggle/input`, so it is "
            "copied to `/kaggle/working` before the canonical notebook runs there. "
            "Locally, this cell is a no-op.",
        ],
    }
    code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            BOOTSTRAP_MARKER + " (injected by scripts/kaggle_notebooks.py)\n",
            "import os\n",
            "import shutil\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            f'BUNDLE = Path("/kaggle/input/{slug}")\n',
            f'WORK = Path("/kaggle/working/{slug}")\n',
            "if BUNDLE.exists():\n",
            "    if not WORK.exists():\n",
            "        shutil.copytree(BUNDLE, WORK)\n",
            "    os.chdir(WORK)\n",
            "    if str(WORK) not in sys.path:\n",
            "        sys.path.insert(0, str(WORK))\n",
            '    print("[kaggle-bootstrap] bundle ready at", WORK)\n',
            "else:\n",
            '    print("[kaggle-bootstrap] /kaggle/input not found; continuing from", Path.cwd())\n',
        ],
    }
    return [markdown, code]


def patch_config_cell(cells: list[dict], args: argparse.Namespace) -> None:
    """Point MOVIELENS_LOCAL_ARCHIVE / AMAZON_LOCAL_SOURCE at Kaggle mounts."""
    replacements = {}
    if args.movielens_input:
        replacements["MOVIELENS_LOCAL_ARCHIVE = \"\""] = (
            f'MOVIELENS_LOCAL_ARCHIVE = "{args.movielens_input}"'
        )
    if args.amazon_input:
        replacements["AMAZON_LOCAL_SOURCE = \"\""] = (
            f'AMAZON_LOCAL_SOURCE = "{args.amazon_input}"'
        )
    if not replacements:
        return
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "MOVIELENS_LOCAL_ARCHIVE" not in source:
            continue
        patched = source
        for old, new in replacements.items():
            if old not in patched:
                raise RuntimeError(f"cannot patch config cell: pattern not found: {old}")
            patched = patched.replace(old, new)
        cell["source"] = patched.splitlines(keepends=True)
        print("[prepare] patched config cell for Kaggle dataset mounts")
        return
    raise RuntimeError("config cell (MOVIELENS_LOCAL_ARCHIVE) not found in notebook")


def build_kaggle_notebook(args: argparse.Namespace) -> dict:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    cells = notebook["cells"]
    # Idempotent bootstrap injection: replace an existing bootstrap, else prepend.
    index = next(
        (
            i
            for i, cell in enumerate(cells)
            if BOOTSTRAP_MARKER in "".join(cell.get("source", []))
        ),
        None,
    )
    bootstrap = bootstrap_cells(args.slug)
    if index is not None:
        cells[index : index + 1] = [bootstrap[1]]
        print("[prepare] replaced existing bootstrap cell")
    else:
        cells[0:0] = bootstrap
    patch_config_cell(cells, args)
    notebook["cells"] = cells
    return notebook


def build_metadata(args: argparse.Namespace, username: str) -> dict:
    return {
        "id": f"{username}/{args.slug}",
        "title": args.title,
        "code_file": KAGGLE_NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": not args.public,
        "enable_gpu": args.gpu,
        "enable_internet": True,
        "dataset_sources": list(args.dataset_slug or []),
        "competition_sources": [],
        "model_sources": [],
    }


def prepare(args: argparse.Namespace, username: str) -> Path:
    if args.clean and BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("actionshap", "configs", "scripts"):
        copy_tree(CODE_ROOT / name, BUNDLE_DIR / name)
    for filename in (
        "requirements.txt",
        "requirements-recommendation.txt",
        "requirements-recommendation.lock",
        "pytest.ini",
    ):
        source = CODE_ROOT / filename
        if source.exists():
            shutil.copy2(source, BUNDLE_DIR / filename)

    notebook = build_kaggle_notebook(args)
    (BUNDLE_DIR / KAGGLE_NOTEBOOK).write_text(json.dumps(notebook, indent=1))
    metadata = build_metadata(args, username)
    (BUNDLE_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"[prepare] bundle ready at {BUNDLE_DIR}")
    print(f"[prepare] kernel id: {metadata['id']}")
    print(f"[prepare] notebook:  {KAGGLE_NOTEBOOK}")
    print(f"[prepare] dataset sources: {metadata['dataset_sources'] or 'none (internet download)'}")
    print(f"[prepare] gpu={metadata['enable_gpu']} internet={metadata['enable_internet']} private={metadata['is_private']}")
    return BUNDLE_DIR


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_push(args: argparse.Namespace) -> int:
    username = read_username()
    bundle = prepare(args, username)
    rc, _, _ = run_cli(
        [args.kaggle_bin, "kernels", "push", "-p", str(bundle)],
        args.dry_run,
        capture=True,
    )
    if rc != 0 and not args.dry_run:
        print("[!] push failed (see error above). The bundle is ready at:")
        print(f"    {bundle}")
        print("    Fix the cause and re-run:  push")
    return rc


def cmd_status(args: argparse.Namespace) -> int:
    username = read_username()
    identity = f"{username}/{args.slug}"
    deadline = time.monotonic() + args.timeout
    while True:
        rc, out, err = run_cli([args.kaggle_bin, "kernels", "status", identity], args.dry_run)
        match = re.search(r"Status:\s*(\S+)", out)
        state = match.group(1) if match else ("error: " + err.strip() if err else "unknown")
        print(f"[{time.strftime('%H:%M:%S')}] {identity}: {state}")
        if args.dry_run:
            return 0
        if rc != 0:
            print("[!] status call failed. If the kernel does not exist yet, run: push")
            return rc
        if state in ("complete", "error", "canceled"):
            print("[status] final state:", state)
            return 0 if state == "complete" else 1
        if time.monotonic() > deadline:
            print(f"[status] still running after {args.timeout}s; poll again with:")
            print(f"    kaggle kernels status {identity}")
            return 2
        time.sleep(args.interval)


def cmd_output(args: argparse.Namespace) -> int:
    username = read_username()
    identity = f"{username}/{args.slug}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rc, _, _ = run_cli(
        [args.kaggle_bin, "kernels", "output", identity, "-p", str(OUTPUT_DIR)],
        args.dry_run,
    )
    if rc == 0:
        print(f"[output] files under {OUTPUT_DIR}:")
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                print(f"    {path.relative_to(OUTPUT_DIR)}  ({path.stat().st_size} bytes)")
    return rc


def cmd_pull(args: argparse.Namespace) -> int:
    username = read_username()
    identity = f"{username}/{args.slug}"
    PULLED_DIR.mkdir(parents=True, exist_ok=True)
    return run_cli(
        [args.kaggle_bin, "kernels", "pull", identity, "-p", str(PULLED_DIR)],
        args.dry_run,
    )[0]


def cmd_list(args: argparse.Namespace) -> int:
    command = [args.kaggle_bin, "kernels", "list", "--page-size", "20"]
    if args.mine or (not args.user and not args.search):
        command += ["--mine"]
    if args.user:
        command += ["--user", args.user]
    if args.search:
        command += ["--search", args.search]
    return run_cli(command, args.dry_run, capture=False)[0]


def cmd_verify(args: argparse.Namespace) -> int:
    ok = True

    config = find_kaggle_json()
    if config is None:
        print("[verify] FAIL  ~/.kaggle/kaggle.json not found")
        ok = False
    else:
        try:
            data = json.loads(config.read_text())
            username, key = data["username"], str(data["key"])
            if len(key) != 32:
                print(f"[verify] WARN  API key length is {len(key)} (expected 32)")
            print(f"[verify] OK    kaggle.json: {config} (username={username})")
        except (KeyError, json.JSONDecodeError) as error:
            print(f"[verify] FAIL  kaggle.json is invalid: {error}")
            ok = False

    try:
        kaggle_bin = find_kaggle_bin()
        print(f"[verify] OK    kaggle CLI: {kaggle_bin}")
    except RuntimeError as error:
        print(f"[verify] FAIL  {error}")
        ok = False
        kaggle_bin = "kaggle"

    if not SOURCE_NOTEBOOK.exists():
        print(f"[verify] FAIL  source notebook missing: {SOURCE_NOTEBOOK}")
        return 1
    json.loads(SOURCE_NOTEBOOK.read_text())
    print(f"[verify] OK    source notebook parses: {SOURCE_NOTEBOOK.name}")

    args.clean = False
    args.kaggle_bin = kaggle_bin
    bundle = prepare(args, read_username() if config is not None else "USERNAME")
    required = [
        "kernel-metadata.json",
        KAGGLE_NOTEBOOK,
        "scripts/run_final_suite.py",
        "scripts/download_datasets.py",
        "configs/final.yaml",
        "actionshap/__init__.py",
        "requirements-recommendation.lock",
    ]
    for relative in required:
        if (bundle / relative).exists():
            print(f"[verify] OK    bundle contains {relative}")
        else:
            print(f"[verify] FAIL  bundle missing {relative}")
            ok = False
    kaggle_notebook = json.loads((bundle / KAGGLE_NOTEBOOK).read_text())
    sources = "".join(
        "".join(cell.get("source", [])) for cell in kaggle_notebook["cells"]
    )
    if BOOTSTRAP_MARKER in sources:
        print(f"[verify] OK    bootstrap cell injected ({BOOTSTRAP_MARKER})")
    else:
        print("[verify] FAIL  bootstrap cell missing")
        ok = False

    print()
    print("[verify] commands that would run (add --dry-run to any subcommand):")
    print(f"    {kaggle_bin} kernels push -p {bundle}")
    print(f"    {kaggle_bin} kernels status {read_username() if config is not None else '<user>'}/{args.slug}")
    print(f"    {kaggle_bin} kernels output <user>/{args.slug} -p {OUTPUT_DIR}")
    print(f"    {kaggle_bin} kernels pull <user>/{args.slug} -p {PULLED_DIR}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push, run, update, and pull ActionShap notebooks on Kaggle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kaggle-bin", default=None, help="Path to the kaggle CLI (default: search PATH).")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle kernel slug (also the bundle/working folder name).")
    parser.add_argument("--username", default=None, help="Kaggle username (default: from kaggle.json).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_options = argparse.ArgumentParser(add_help=False)
    common_options.add_argument("--dry-run", action="store_true",
                                help="Print commands without executing them.")

    bundle_options = argparse.ArgumentParser(add_help=False)
    bundle_options.add_argument("--clean", action="store_true", help="Rebuild the bundle from scratch.")
    bundle_options.add_argument("--public", action="store_true", help="Make the kernel public (default: private).")
    bundle_options.add_argument("--gpu", action="store_true", help="Enable GPU (default: CPU; this suite is CPU-only).")
    bundle_options.add_argument("--title", default=DEFAULT_TITLE, help="Kernel title shown on Kaggle.")
    bundle_options.add_argument("--dataset-slug", action="append", default=[], metavar="OWNER/SLUG",
                                help="Mount a Kaggle dataset (repeatable).")
    bundle_options.add_argument("--movielens-input", default="", metavar="PATH",
                                help="/kaggle/input/... path to the ml-1m.zip inside a mounted dataset.")
    bundle_options.add_argument("--amazon-input", default="", metavar="PATH",
                                help="/kaggle/input/... path to Digital_Music_5.json.gz inside a mounted dataset.")

    sub.add_parser("prepare", parents=[bundle_options, common_options], help="Assemble the bundle offline (no network).")
    sub.add_parser("verify", parents=[bundle_options, common_options], help="Validate credentials, CLI, notebook, and bundle locally.")

    p_push = sub.add_parser("push", parents=[bundle_options, common_options],
                            help="Prepare, upload, and start a cloud run (re-push = update).")

    p_status = sub.add_parser("status", parents=[common_options], help="Poll the run until complete / error / canceled.")
    p_status.add_argument("--interval", type=int, default=60, help="Poll interval in seconds.")
    p_status.add_argument("--timeout", type=int, default=6 * 3600, help="Give up after this many seconds.")

    sub.add_parser("output", parents=[common_options], help="Download the run's output files to code/kaggle/output/.")
    sub.add_parser("pull", parents=[common_options], help="Download the executed notebook to code/kaggle/pulled/.")
    p_list = sub.add_parser("list", parents=[common_options], help="List notebooks on Kaggle.")
    p_list.add_argument("--user", default=None, help="List a user's public notebooks (default: your own, including private).")
    p_list.add_argument("--search", default=None, help="Search public notebooks by query (e.g. 'recommendation').")
    p_list.add_argument("--mine", action="store_true", help="Explicitly list your own notebooks.")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.kaggle_bin:
            args.kaggle_bin = str(Path(args.kaggle_bin).expanduser())
        else:
            args.kaggle_bin = find_kaggle_bin()
        if args.username:
            pass  # resolved inside commands that need it
        handlers = {
            "prepare": lambda a: prepare(a, read_username()),
            "verify": cmd_verify,
            "push": cmd_push,
            "status": cmd_status,
            "output": cmd_output,
            "pull": cmd_pull,
            "list": cmd_list,
        }
        return int(handlers[args.command](args) or 0)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[!] {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
