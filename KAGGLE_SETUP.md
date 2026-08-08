# Linking Kaggle to this environment

## Status (checked 2026-08-08)

| Piece | State |
|---|---|
| `kaggle` CLI (v2.2.4) | ✅ installed in `/home/user/kaggle-venv` |
| `kagglehub` (v1.0.2) | ✅ installed in `/home/user/kaggle-venv` |
| Network to `api.kaggle.com` | ✅ verified reachable from this sandbox |
| API credentials (`kaggle.json`) | ⏳ missing — this is the only remaining step |

## Step 1 — Get your API token (30 seconds, on your machine)

1. Sign in at **https://www.kaggle.com**
2. Click your avatar (top right) → **Settings**
3. Scroll to the **API** section → click **Create New Token**
4. Kaggle downloads a file named **`kaggle.json`** containing:

```json
{"username":"your_kaggle_username","key":"a_40_char_api_key"}
```

## Step 2 — Link it to this sandbox

Pick one of these two ways (option A is easiest):

- **A. Upload the file here:** attach `kaggle.json` to your next message in this chat. I will place it at `~/.kaggle/kaggle.json`, set permissions to `600`, and verify the link with a real API call.
- **B. Environment variables:** set `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment (e.g. in Arena's environment settings), then run:
  ```bash
  source /home/user/kaggle-venv/bin/activate
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_api_key
  ```

> ⚠️ Never paste the API key into the chat as plain text. The file upload (option A) keeps it out of the conversation.
> Keep `kaggle.json` private — anyone with it can use your Kaggle account and quota.

## Step 3 — Verify (I'll run this once credentials are in)

```bash
source /home/user/kaggle-venv/bin/activate
kaggle datasets list -s movielens      # should print a table, not "Authentication required"
```

## Daily-use cheat sheet

```bash
# activate the env once per session
source /home/user/kaggle-venv/bin/activate

# search / download datasets
kaggle datasets list -s "movie lens"
kaggle datasets download -d prajitdatta/movielens-1m-dataset
kaggle datasets download -d saurav9786/amazon-digital-music-reviews

# list files inside a dataset without downloading
kaggle datasets files prajitdatta/movielens-1m-dataset

# upload / manage your own datasets
kaggle datasets create -p ./my-dataset
kaggle datasets status <owner>/<dataset>

# competitions
kaggle competitions list
kaggle competitions download -c <competition-name>

# notebooks (Kaggle "kernels")
kaggle kernels list --user <username>
kaggle kernels pull <owner>/<kernel-name>

# in Python (kagglehub, the modern library)
import kagglehub
path = kagglehub.dataset_download("prajitdatta/movielens-1m-dataset")
print(path)   # local folder with the downloaded files
```

## Bonus: fits your ActionShap repo

Your `next-paper` pipeline downloads MovieLens-1M (GroupLens) and Amazon
Digital Music (UCSD) directly — and the docs note those mirrors are often
blocked. Both datasets exist on Kaggle. Once the link works, I can add Kaggle
as a fallback source in `paper-ideas/ActionShap/code/scripts/download_datasets.py`
(with the same SHA-256 provenance checks). Just say the word.

> **Update:** Kaggle notebook execution tooling now lives in the repo — see
> `paper-ideas/ActionShap/KAGGLE_NOTEBOOKS.md` (push/run/update/pull the
> ActionShap notebook via the Kaggle API).
