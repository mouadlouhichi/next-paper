#!/usr/bin/env bash
# Round-7 sandbox orchestrator, wave 2: multi-seed design ablations (R7-2) and
# remaining diagnostics. Waits for wave 1 (run_sandbox_round7.sh) to finish.
set -u
cd "$(dirname "$0")/.."
LOGDIR=results/journal_runs/_notebook_logs
mkdir -p "$LOGDIR"
export COALGAME_DEVICE=cpu
export COALGAME_THREADS=2
export PYTHONUNBUFFERED=1
PY=python3

echo "wave2: waiting for wave1 to finish..."
while pgrep -f "run_sandbox_round7.sh" > /dev/null; do sleep 30; done
echo "wave2: wave1 done, starting ($(date))"

stage() {
  echo "======================================================================"
  echo "STAGE: $1  ($(date))"
  echo "======================================================================"
}

stage "w2-1/4 design ablations amazon seed 43 (full users)"
$PY scripts/run_design_ablations.py --dataset amazon --seed 43 \
  > "$LOGDIR/sandbox_design_ablations_amazon_43.log" 2>&1
echo "w2 stage1 exit=$?"

stage "w2-2/4 design ablations amazon seed 44 (full users)"
$PY scripts/run_design_ablations.py --dataset amazon --seed 44 \
  > "$LOGDIR/sandbox_design_ablations_amazon_44.log" 2>&1
echo "w2 stage2 exit=$?"

stage "w2-3/4 negset sensitivity ml1m (1500-user documented subsample)"
$PY scripts/run_negset_sensitivity.py --dataset ml1m \
  --source-run results/journal_runs/ml1m_lightgcn_v3_prospective \
  --out results/journal_runs/ml1m_lightgcn_v6_negset_sensitivity \
  --sizes 50 100 500 --max-users 1500 \
  > "$LOGDIR/sandbox_negset_ml1m.log" 2>&1
echo "w2 stage3 exit=$?"

stage "w2-4/4 masked-forward ml1m seed 44"
$PY scripts/run_masked_forward_faithfulness.py --dataset ml1m --seed 44 --n-users 1000 \
  > "$LOGDIR/sandbox_maskedfwd_ml1m_44.log" 2>&1
echo "w2 stage4 exit=$?"

echo "WAVE2 ALL DONE ($(date))"
