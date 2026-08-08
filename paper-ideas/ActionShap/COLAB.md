# Running ActionShap Notebooks on Google Colab

Colab has **no public "push & run" API** like Kaggle's — free Colab is driven
from the browser. The standard free workflow is **one click in Colab that
loads the notebook from GitHub and runs it on Google's hardware**, and the
paid option (Colab Enterprise) has a real execution API. Both are covered
here.

```
┌──────────────────┐  git push   ┌─────────────────────────────┐
│  this repository │ ──────────► │  github.com/mouadlouhichi/  │
│  colab_notebooks │             │  next-paper (branch)        │
│  .py push        │             └──────────────┬──────────────┘
│                  │                            │ one click
│  your browser ◄──┼────────────────────────────▼──────────────┐
│  colab.research. │   clone repo → run every cell on Colab    │
│  google.com      │   (free CPU/GPU, your Google account)     │
└──────────────────┘                                            │
   Colab Enterprise (paid, GCP): NotebookExecutionJob API       │
   submit from GCS → poll → download outputs                    │
└───────────────────────────────────────────────────────────────┘
```

## The one-click link (free)

```bash
python scripts/colab_notebooks.py push
```

This regenerates `code/colab/ActionShap_All_colab.ipynb` (the canonical
notebook **plus a Colab bootstrap cell**), commits it, pushes the current
branch, and prints:

```
https://colab.research.google.com/github/mouadlouhichi/next-paper/blob/
  <branch>/paper-ideas/ActionShap/code/colab/ActionShap_All_colab.ipynb
```

Click it, sign in with your Google account, and choose **Runtime → Run all**.
The bootstrap cell clones the repository into `/content/next-paper` and
changes into the code directory — required because Colab opens only the
notebook file, while the `actionshap` package, configs, and scripts live in
the repo. Everything else runs exactly like the local Run-All notebook
(dependency install, dataset download, 85-command suite, validation,
packaging).

### Updating

Edit `ActionShap_All.ipynb` (or configs/scripts), then run `push` again —
same link, new version, run it again. That is the whole update loop.

## Commands

| Command | What it does |
|---|---|
| `python scripts/colab_notebooks.py push` | Regenerate + commit + push the branch, print the link |
| `python scripts/colab_notebooks.py url` | Print the link and badge markdown only |
| `python scripts/colab_notebooks.py prepare` | Regenerate the Colab notebook locally (no git) |
| `python scripts/colab_notebooks.py verify` | Local sanity checks (no network) |
| `python scripts/colab_notebooks.py drive --credentials credentials.json` | Upload to Google Drive, print an Open-with-Colab link |
| `python scripts/colab_notebooks.py enterprise` | Explain the paid Colab Enterprise API |

Options: `--branch <name>` (default: current branch), `--dry-run`.

## Badge (add to README)

```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/mouadlouhichi/next-paper/blob/
<branch>/paper-ideas/ActionShap/code/colab/ActionShap_All_colab.ipynb)
```

## Optional: Google Drive upload

The `drive` command pushes the notebook to your Drive and returns an
Open-with-Colab link. It needs your own Google OAuth client:

```bash
pip install google-auth-oauthlib google-api-python-client
# console.cloud.google.com → APIs & Services → Credentials →
#   Create credentials → OAuth client ID → Desktop app → download JSON
python scripts/colab_notebooks.py drive --credentials credentials.json
```

First run opens a browser for consent. Note: Drive upload is only useful as
an alternative to the GitHub path — the GitHub path needs no OAuth at all.

## Paid: Colab Enterprise execution API

If you need true headless push/run/poll/output (Kaggle-style) on Colab, the
official way is the Colab Enterprise API — a `NotebookExecutionJob` submitted
from Cloud Storage. It requires a GCP project with billing. See
`python scripts/colab_notebooks.py enterprise` for the REST outline, or the
official reference: https://cloud.google.com/colab/docs/reference/rest

## Caveats

* **This Arena sandbox cannot reach Google** (colab.research.google.com,
  googleapis.com, drive.google.com are blocked at the network layer), so the
  `drive` and `enterprise` steps must be run from your own machine.
  `prepare` / `url` / `push` / `verify` only need git + GitHub and work here.
* The branch you open in Colab must be pushed to GitHub (`push` does that).
* Colab sessions are ephemeral: after closing the tab, results live in
  `/content/next-paper/paper-ideas/ActionShap/code/paper/final/...` and are
  lost unless you download them (Files panel → download) or mount Drive.
* Free Colab disconnects after long idle/runtime; the ~full ActionShap suite
  is CPU-heavy — GPU is not needed (CPU-only suite), but keep the tab open.
