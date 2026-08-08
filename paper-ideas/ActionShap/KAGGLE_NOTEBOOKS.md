# Running ActionShap Notebooks on Kaggle (push · run · update · pull)

The canonical `ActionShap_All.ipynb` Run-All notebook can be executed on
Kaggle's cloud via the Kaggle API. One command uploads the notebook **plus
the `actionshap` package, configs, and scripts**, starts the run, and every
re-push **updates** the same kernel and triggers a fresh run.

```
┌──────────────────┐   push    ┌───────────────────────────┐
│  your machine    │ ────────► │  Kaggle cloud (CPU/GPU)   │
│  kaggle_notebooks│           │  /kaggle/input/...  (ro)  │
│  .py push        │           │      │ copytree           │
│                  │           │      ▼                    │
│  status ─────────┼──────────►│  /kaggle/working/... (rw) │
│  output ─────────┼──────────►│  notebook runs everything │
│  pull   ◄────────┼───────────│  outputs + executed .ipynb│
└──────────────────┘           └───────────────────────────┘
```

## Prerequisites

1. **`kaggle` CLI** — `pip install kaggle`
2. **Credentials** — `~/.kaggle/kaggle.json` (username + 32-char API key,
   `chmod 600`). Get a token at kaggle.com → avatar → Settings → API →
   *Create New Token*.
3. **Network access to `api.kaggle.com`.** The Arena sandbox currently blocks
   Kaggle egress (TLS connection is killed), so run these commands from a
   machine that can reach Kaggle — or ask the platform to open Kaggle egress.
   Everything except the actual API calls can be tested offline (`verify`,
   `prepare`, `--dry-run`).

## Workflow

Run all commands from `paper-ideas/ActionShap/code`:

```bash
# 0. Sanity-check everything without touching the network
python scripts/kaggle_notebooks.py verify

# 1. Build the bundle, upload it, and start the cloud run
python scripts/kaggle_notebooks.py push

# 2. Watch the run (polls until complete / error / canceled)
python scripts/kaggle_notebooks.py status --interval 60

# 3. When complete: download outputs (results, assets, manifests)
python scripts/kaggle_notebooks.py output

# 4. Download the executed notebook (outputs embedded) for inspection
python scripts/kaggle_notebooks.py pull
```

### Updating the notebook (the "update" workflow)

Edit `ActionShap_All.ipynb` (or the configs/scripts in `code/`), then push
again — same command, same kernel id, new version, new run:

```bash
python scripts/kaggle_notebooks.py push
python scripts/kaggle_notebooks.py status
```

### Options

| Flag | Effect |
|---|---|
| `--dry-run` | Print the exact `kaggle` commands without executing them |
| `--clean` | Rebuild the bundle from scratch (no stale files) |
| `--public` | Make the kernel public (default: private) |
| `--gpu` | Enable GPU (default CPU — this suite is CPU-only) |
| `--slug NAME` | Kernel slug; default `actionshap-rev4`. If you change it, also change `--title` so the title slugifies to it |
| `--title TEXT` | Kernel title shown on Kaggle |
| `--interval SEC`, `--timeout SEC` | `status` polling |

### Using Kaggle-hosted dataset mirrors (skip the download)

By default the notebook downloads MovieLens-1M and Amazon Digital Music from
the GroupLens/UCSD mirrors using Kaggle's internet egress. To mount Kaggle
mirrors instead, first find datasets that contain the exact source files
(e.g. `ml-1m.zip` and `Digital_Music_5.json.gz`), then:

```bash
python scripts/kaggle_notebooks.py push \
  --dataset-slug <owner>/<movielens-dataset> \
  --movielens-input /kaggle/input/<movielens-dataset>/ml-1m.zip \
  --amazon-input /kaggle/input/<amazon-dataset>/Digital_Music_5.json.gz
```

`--dataset-slug` mounts the dataset (repeatable) and the `--*-input` flags
point the downloader at the mounted files. The pipeline keeps its SHA-256
provenance checks either way.

## What gets pushed

`code/kaggle/bundle/` (rebuilt on every push):

```
bundle/
├── kernel-metadata.json          # id, title, gpu/internet flags, dataset sources
├── ActionShap_All_kaggle.ipynb   # your notebook + injected Kaggle bootstrap
├── actionshap/                   # the package
├── configs/final.yaml            # frozen experiment contract
├── scripts/                      # downloaders, suite, validation, packaging
└── requirements*.txt / .lock     # locked environment
```

Two automatic adaptations, both idempotent:

1. **Bootstrap cell** (prepended): Kaggle's `/kaggle/input/<slug>` is
   read-only, so the bundle is copied to `/kaggle/working/<slug>` and the
   notebook runs from there — otherwise the suite could not write results.
2. **Config-cell patch** (only with `--movielens-input` / `--amazon-input`):
   `MOVIELENS_LOCAL_ARCHIVE` / `AMAZON_LOCAL_SOURCE` are pointed at the
   mounted files.

## Where results land

- `code/kaggle/output/` — `kaggle kernels output` download (results, paper
  assets, manifests, release package).
- `code/kaggle/pulled/` — the executed notebook with all outputs embedded.
- Kaggle UI — kaggle.com → your profile → **Code** → `actionshap-rev4`
  (live logs, kernel output, version history).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `SSLZeroReturnError` / `TLS/SSL connection has been closed` | This machine cannot reach Kaggle (Arena sandbox blocks egress). Run from your own machine or ask the platform to allow `api.kaggle.com`. |
| `401 Unauthorized` | API key invalid/rotated. Create a new token and update `~/.kaggle/kaggle.json`. |
| `Your kernel title does not resolve to the specified id` | `--title` must slugify to `--slug` (e.g. "ActionShap Rev4" → `actionshap-rev4`). |
| Status ends in `error` | Open the kernel on kaggle.com → *Output* / *Logs* tabs for the traceback. Common: dependency resolution (check the lock file) or dataset download failures. |
| Kernel exists but `push` seems to do nothing | Re-push always creates a new version and a new run; check the *Versions* tab in the Kaggle UI. |
