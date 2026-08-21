#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
mkdir -p results/journal_runs/_notebook_logs
export COALGAME_DEVICE=cpu COALGAME_THREADS=2 PYTHONUNBUFFERED=1
echo "CHAIN START ($(date))"

echo "=== [1/4] v7 corrected protocol (amazon) ($(date))"
C1_WITH_SHAPLEY=1 bash scripts/run_and_commit.sh amazon_books_lightgcn_v7_corrected_protocol \
  python3 scripts/run_protocol_v7.py --dataset amazon \
  --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \
  --out results/journal_runs/amazon_books_lightgcn_v7_corrected_protocol \
  > results/journal_runs/_notebook_logs/sandbox_v7_amazon.log 2>&1
echo "[1/4] done rc=$? ($(date))"

echo "=== [2/4] controlled randomization (amazon) ($(date))"
bash scripts/run_and_commit.sh amazon_books_lightgcn_v8_controlled_randomization \
  python3 scripts/run_controlled_randomization.py --dataset amazon \
  --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \
  --out results/journal_runs/amazon_books_lightgcn_v8_controlled_randomization \
  > results/journal_runs/_notebook_logs/sandbox_ctrlrand_amazon.log 2>&1
echo "[2/4] done rc=$? ($(date))"

echo "=== [3/4] v7 corrected protocol (ml1m) ($(date))"
C1_WITH_SHAPLEY=1 bash scripts/run_and_commit.sh ml1m_lightgcn_v7_corrected_protocol \
  python3 scripts/run_protocol_v7.py --dataset ml1m \
  --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
  --out results/journal_runs/ml1m_lightgcn_v7_corrected_protocol \
  > results/journal_runs/_notebook_logs/sandbox_v7_ml1m.log 2>&1
echo "[3/4] done rc=$? ($(date))"

echo "=== [4/4] sequential baselines (amazon, 3 seeds) ($(date))"
bash scripts/run_and_commit.sh amazon_books_lightgcn_v8_sequential_baselines \
  python3 scripts/run_sequential_baselines.py --dataset amazon \
  --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \
  --out results/journal_runs/amazon_books_lightgcn_v8_sequential_baselines \
  --seeds 42 43 44 \
  > results/journal_runs/_notebook_logs/sandbox_seqbaselines_amazon.log 2>&1
echo "[4/4] done rc=$? ($(date))"

echo "CHAIN DONE ($(date))"
