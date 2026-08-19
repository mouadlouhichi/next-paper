#!/usr/bin/env bash
# Round-7 sandbox orchestrator: runs the remaining R7-2/R7-3 experiments on CPU
# (2 cores). Stages are ordered by reviewer value; each stage is independently
# re-runnable and writes its own log + artifacts.
set -u
cd "$(dirname "$0")/.."
LOGDIR=results/journal_runs/_notebook_logs
mkdir -p "$LOGDIR"
export COALGAME_DEVICE=cpu
export COALGAME_THREADS=2
export PYTHONUNBUFFERED=1
PY=python3

stage() {
  echo "======================================================================"
  echo "STAGE: $1  ($(date))"
  echo "======================================================================"
}

stage "1/6 randomization sanity (amazon)"
$PY scripts/run_randomization_sanity.py --dataset amazon \
  --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \
  --out results/journal_runs/amazon_books_lightgcn_v6_randomization_sanity \
  > "$LOGDIR/sandbox_randomization_amazon.log" 2>&1
echo "stage1 exit=$?"

stage "2/6 negset sensitivity amazon (1500-user subsample, documented)"
$PY scripts/run_negset_sensitivity.py --dataset amazon \
  --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \
  --out results/journal_runs/amazon_books_lightgcn_v6_negset_sensitivity \
  --sizes 50 100 500 --max-users 1500 \
  > "$LOGDIR/sandbox_negset_amazon.log" 2>&1
echo "stage6 exit=$?"

echo "ALL STAGES DONE ($(date))"
stage "3/6 masked-forward amazon seed 43"
$PY scripts/run_masked_forward_faithfulness.py --dataset amazon --seed 43 --n-users 1000 \
  > "$LOGDIR/sandbox_maskedfwd_amazon_43.log" 2>&1
echo "stage3 exit=$?"

stage "4/6 masked-forward amazon seed 44"
$PY scripts/run_masked_forward_faithfulness.py --dataset amazon --seed 44 --n-users 1000 \
  > "$LOGDIR/sandbox_maskedfwd_amazon_44.log" 2>&1
echo "stage4 exit=$?"

stage "5/6 randomization sanity (ml1m)"
$PY scripts/run_randomization_sanity.py --dataset ml1m \
  --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
  --out results/journal_runs/ml1m_lightgcn_v6_randomization_sanity \
  > "$LOGDIR/sandbox_randomization_ml1m.log" 2>&1
echo "stage2 exit=$?"

stage "6/6 masked-forward ml1m seed 43"
$PY scripts/run_masked_forward_faithfulness.py --dataset ml1m --seed 43 --n-users 1000 \
  > "$LOGDIR/sandbox_maskedfwd_ml1m_43.log" 2>&1
echo "stage5 exit=$?"

