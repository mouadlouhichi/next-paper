#!/usr/bin/env python3
"""Push, run, and update ActionShap notebooks on Google Colab.

Colab has no public push/run API like Kaggle's. This tool implements the
practical, working alternatives:

    prepare   Inject a Colab bootstrap cell (repo clone + chdir) into a
              Colab-ready copy of the ActionShap notebook.
    url       Print the one-click "Open in Colab" link for the GitHub copy.
    push      prepare + commit the Colab notebook + push the current branch,
              then print the link (the "update" workflow is: edit -> push).
    drive     Upload the notebook to Google Drive and print an "Open with
              Colab" link (needs your own Google OAuth client, see below).
    enterprise  Explain the paid Colab Enterprise execution API (no code runs).
    verify    Local sanity checks, no network required.

The bootstrap cell is required because "Open in Colab" loads ONLY the
notebook file - the actionshap package, configs, and scripts live in the
repo, so the cell clones the repository into /content/next-paper and changes
into the code directory before the canonical Run-All cells execute.

Prerequisites for the GitHub path
---------------------------------
* A GitHub repository containing this project (the notebook is opened via
  https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/...).
* The branch you open must be pushed to GitHub.

Optional: Google Drive upload
-----------------------------
* pip install google-auth-oauthlib google-api-python-client
* Create an OAuth Desktop client at console.cloud.google.com, download
  credentials.json, and pass --credentials credentials.json. The first run
  opens a browser for consent. This only works on a machine with a browser.

NOTE: the Arena sandbox cannot reach Google (colab.research.google.com,
googleapis.com, drive.google.com are blocked at the network layer). Run the
drive/enterprise steps from your own machine; prepare/url/push/verify work
here because they only need git and GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # paper-ideas/ActionShap
CODE_ROOT = REPO_ROOT / "code"
COLAB_DIR = CODE_ROOT / "colab"
SOURCE_NOTEBOOK = CODE_ROOT / "ActionShap_All.ipynb"
COLAB_NOTEBOOK = COLAB_DIR / "ActionShap_All_colab.ipynb"
BOOTSTRAP_MARKER = "# ==== COLAB BOOTSTRAP"
DEFAULT_REPO = "https://github.com/mouadlouhichi/next-paper.git"


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def git(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *command], cwd=cwd, capture_output=True, text=True)


def current_branch() -> str:
    proc = git(["branch", "--show-current"])
    branch = proc.stdout.strip()
    if not branch:
        raise RuntimeError("not on a git branch; cannot build a Colab URL")
    return branch


def repo_url() -> str:
    proc = git(["remote", "get-url", "origin"])
    url = proc.stdout.strip()
    if not url:
        return DEFAULT_REPO
    url = url.removesuffix(".git")
    if url.startswith("git@"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url + ".git"


def repo_owner_name() -> tuple[str, str]:
    url = repo_url().removesuffix(".git")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+)$", url)
    if not match:
        raise RuntimeError(f"cannot parse GitHub owner/repo from: {url}")
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# Notebook preparation
# ---------------------------------------------------------------------------

def bootstrap_cell(branch: str) -> list[dict]:
    markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Colab cloud run\n",
            "This notebook was prepared by `scripts/colab_notebooks.py`. Colab "
            "opens only this file, so the bootstrap cell clones the repository "
            "into `/content/next-paper` and changes into the code directory "
            "before the canonical Run-All cells execute.",
        ],
    }
    code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            BOOTSTRAP_MARKER + " (injected by scripts/colab_notebooks.py)\n",
            "import os\n",
            "import subprocess\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            f'REPO_URL = "{repo_url()}"\n',
            f'BRANCH = "{branch}"\n',
            'REPO_DIR = Path("/content/next-paper")\n',
            "if not (REPO_DIR / \".git\").exists():\n",
            "    subprocess.run(\n",
            '        ["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO_DIR)],\n',
            "        check=True,\n",
            "    )\n",
            'CODE_ROOT = REPO_DIR / "paper-ideas" / "ActionShap" / "code"\n',
            "os.chdir(CODE_ROOT)\n",
            "if str(CODE_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(CODE_ROOT))\n",
            'print("[colab-bootstrap] repo ready at", CODE_ROOT)\n',
        ],
    }
    return [markdown, code]


def prepare(branch: str) -> Path:
    if not SOURCE_NOTEBOOK.exists():
        raise RuntimeError(f"source notebook missing: {SOURCE_NOTEBOOK}")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    cells = notebook["cells"]
    index = next(
        (
            i
            for i, cell in enumerate(cells)
            if BOOTSTRAP_MARKER in "".join(cell.get("source", []))
        ),
        None,
    )
    bootstrap = bootstrap_cell(branch)
    if index is not None:
        cells[index : index + 1] = [bootstrap[1]]
        print("[prepare] replaced existing bootstrap cell")
    else:
        cells[0:0] = bootstrap
    notebook["cells"] = cells
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    COLAB_NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    print(f"[prepare] Colab notebook written to {COLAB_NOTEBOOK}")
    return COLAB_NOTEBOOK


def colab_url(branch: str) -> str:
    owner, repo = repo_owner_name()
    relative = COLAB_NOTEBOOK.relative_to(REPO_ROOT.parent.parent)
    return (
        "https://colab.research.google.com/github/"
        f"{owner}/{repo}/blob/{branch}/{relative}"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    branch = args.branch or current_branch()
    prepare(branch)
    print(f"\n[prepare] one-click link (after pushing):\n    {colab_url(branch)}")
    return 0


def cmd_url(args: argparse.Namespace) -> int:
    branch = args.branch or current_branch()
    print(colab_url(branch))
    print("\nBadge markdown:")
    print(
        "[![Open in Colab]"
        "(https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url(branch)})"
    )
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    branch = args.branch or current_branch()
    notebook = prepare(branch)
    if args.dry_run:
        print(f"$ git add {notebook.relative_to(REPO_ROOT.parent.parent)}")
        print(f"$ git commit -m 'Regenerate Colab notebook'")
        print(f"$ git push origin {branch}")
        print(f"\n[push] one-click link:\n    {colab_url(branch)}")
        return 0
    git(["add", str(notebook)])
    proc = git(["commit", "-m", "Regenerate Colab notebook ({})".format(branch)])
    if proc.returncode not in (0, 1):  # 1 = nothing to commit
        print(proc.stderr)
        return proc.returncode
    proc = git(["push", "origin", branch])
    if proc.returncode != 0:
        print(proc.stderr)
        return proc.returncode
    print(f"[push] committed and pushed to origin/{branch}")
    print(f"[push] one-click link:\n    {colab_url(branch)}")
    return 0


def cmd_drive(args: argparse.Namespace) -> int:
    try:
        from google.auth.transport.requests import Request  # noqa: F401
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as error:
        print(
            "[!] Google client libraries are missing. Install them with:\n"
            "    pip install google-auth-oauthlib google-api-python-client"
        )
        return 1
    credentials = Path(args.credentials).expanduser()
    if not credentials.exists():
        print(
            f"[!] credentials file not found: {credentials}\n"
            "    Create an OAuth Desktop client at console.cloud.google.com "
            "and download credentials.json."
        )
        return 1
    branch = args.branch or current_branch()
    notebook = prepare(branch)
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials), ["https://www.googleapis.com/auth/drive.file"]
    )
    creds = flow.run_local_server(port=0, open_browser=True)
    service = build("drive", "v3", credentials=creds)
    media = MediaFileUpload(
        str(notebook), mimetype="application/json", resumable=True
    )
    body = {
        "name": notebook.name,
        "mimeType": "application/vnd.google.colaboratory",
    }
    uploaded = service.files().create(body=body, media_body=media, fields="id").execute()
    print(f"[drive] uploaded: {notebook.name}")
    print(f"[drive] open in Colab: https://colab.research.google.com/drive/{uploaded['id']}")
    return 0


def cmd_enterprise(args: argparse.Namespace) -> int:
    print(
        "Colab Enterprise execution API (paid, requires GCP)\n"
        "--------------------------------------------------\n"
        "Free Colab has no headless execution API. Google's official one is the\n"
        "Colab Enterprise API - a NotebookExecutionJob submitted from GCS:\n"
        "\n"
        "  1. Create a GCP project with billing + Colab Enterprise enabled.\n"
        "  2. Upload the prepared notebook to Cloud Storage:\n"
        "       gsutil cp colab/ActionShap_All_colab.ipynb gs://YOUR_BUCKET/\n"
        "  3. Submit an execution (REST):\n"
        "       POST https://colab.googleapis.com/v1/projects/{project}/locations/"
        "{region}/notebookExecutionJobs\n"
        "       {\"gcsNotebookSource\": {\"uri\": \"gs://YOUR_BUCKET/..."
        "          .ipynb\"},\n"
        "        \"notebookRuntimeTemplateResourceName\": \"...\",\n"
        "        \"executionTimeout\": \"3600s\"}\n"
        "  4. Poll GET .../notebookExecutionJobs/{id} until SUCCEEDED, then\n"
        "     download the output notebook from the job's GCS output directory.\n"
        "\n"
        "Full reference: https://cloud.google.com/colab/docs/reference/rest\n"
        "Free alternative: use `url` / `push` (one click in your browser)."
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ok = True
    try:
        branch = current_branch()
        print(f"[verify] OK    current branch: {branch}")
    except RuntimeError as error:
        print(f"[verify] FAIL  {error}")
        ok = False
        branch = "main"
    try:
        owner, repo = repo_owner_name()
        print(f"[verify] OK    GitHub repo: {owner}/{repo}")
    except RuntimeError as error:
        print(f"[verify] FAIL  {error}")
        ok = False
    try:
        notebook = prepare(branch)
        content = notebook.read_text()
        if BOOTSTRAP_MARKER in content and "REPO_URL" in content:
            print(f"[verify] OK    bootstrap cell injected in {notebook.name}")
        else:
            print("[verify] FAIL  bootstrap cell missing")
            ok = False
        print(f"[verify] OK    Colab notebook parses: {notebook}")
        print(f"[verify] OK    one-click URL: {colab_url(branch)}")
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"[verify] FAIL  {error}")
        ok = False
    print()
    print("[verify] next steps:")
    print("    python scripts/colab_notebooks.py push   # commit + push + print link")
    print("    python scripts/colab_notebooks.py url    # print link only")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push, run, and update ActionShap notebooks on Google Colab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--branch", default=None, help="Git branch for the Colab link (default: current).")
    sub = parser.add_subparsers(dest="command", required=True)

    common_options = argparse.ArgumentParser(add_help=False)
    common_options.add_argument("--dry-run", action="store_true",
                                help="Print commands without executing them.")

    sub.add_parser("prepare", parents=[common_options], help="Inject the Colab bootstrap and write the Colab-ready notebook.")
    sub.add_parser("url", parents=[common_options], help="Print the one-click Open-in-Colab link and badge.")
    sub.add_parser("push", parents=[common_options], help="prepare + commit + push the branch, then print the link.")
    p_drive = sub.add_parser("drive", parents=[common_options], help="Upload the notebook to Google Drive (needs OAuth credentials).")
    p_drive.add_argument("--credentials", required=True, help="Path to credentials.json (Google OAuth Desktop client).")
    sub.add_parser("enterprise", parents=[common_options], help="Explain the paid Colab Enterprise execution API.")
    sub.add_parser("verify", parents=[common_options], help="Local sanity checks (no network).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        handlers = {
            "prepare": cmd_prepare,
            "url": cmd_url,
            "push": cmd_push,
            "drive": cmd_drive,
            "enterprise": cmd_enterprise,
            "verify": cmd_verify,
        }
        return int(handlers[args.command](args) or 0)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"[!] {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
