#!/usr/bin/env bash
# FULL QUEUE: every released round-9 experiment, ordered by reviewer value,
# with skip-guards and periodic auto-commits. Resume-safe at every level.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/journal_runs/_notebook_logs
export COALGAME_DEVICE=cpu COALGAME_THREADS=2 PYTHONUNBUFFERED=1
if ! python3 -c "import numpy, scipy, pandas, torch" 2>/dev/null; then
  echo "deps missing, reinstalling..."
  pip install --break-system-packages -q numpy scipy pandas tqdm pyyaml pyarrow requests torch matplotlib || \
  pip install -q numpy scipy pandas tqdm pyyaml pyarrow requests torch matplotlib
fi
echo "FULL-QUEUE START ($(date))"

(
  while true; do
    sleep 600
    cd /home/user/next-paper || exit
    git add -A paper-ideas/CoalGameRec/code/results/journal_runs 2>/dev/null || true
    git -c user.email="arena-agent@users.noreply.github.com" -c user.name="arena-agent" \
        commit -q -m "periodic checkpoint of in-flight run artifacts ($(date -u +%H:%M))" 2>/dev/null || true
    git push -q origin arena/019fdd75-next-paper 2>/dev/null || true
  done
) &
COMMITTER_PID=$!
trap "kill $COMMITTER_PID 2>/dev/null" EXIT

run() {
  NAME="$1"; OUT="$2"; shift 2
  echo "=== [$NAME] start ($(date))"
  bash scripts/run_and_commit.sh "$OUT" "$@" \
    > "results/journal_runs/_notebook_logs/fullq_${NAME}.log" 2>&1
  echo "=== [$NAME] done rc=$? ($(date))"
}

V3M=results/journal_runs/ml1m_lightgcn_v3_prospective
V3A=results/journal_runs/amazon_books_lightgcn_v3_prospective

# 1. corrected protocol (v7)
if [ ! -f results/journal_runs/amazon_books_lightgcn_v7_corrected_protocol/tables/summary_mean_std.csv ]; then
  run v7_amazon amazon_books_lightgcn_v7_corrected_protocol \
    env C1_WITH_SHAPLEY=1 python3 scripts/run_protocol_v7.py --dataset amazon \
    --source-run $V3A --out results/journal_runs/amazon_books_lightgcn_v7_corrected_protocol
fi
if [ ! -f results/journal_runs/ml1m_lightgcn_v7_corrected_protocol/tables/summary_mean_std.csv ]; then
  run v7_ml1m ml1m_lightgcn_v7_corrected_protocol \
    env C1_WITH_SHAPLEY=1 python3 scripts/run_protocol_v7.py --dataset ml1m \
    --source-run $V3M --out results/journal_runs/ml1m_lightgcn_v7_corrected_protocol
fi

# 2. controlled randomization
run ctrlrand_amazon amazon_books_lightgcn_v8_controlled_randomization \
  python3 scripts/run_controlled_randomization.py --dataset amazon --source-run $V3A \
  --out results/journal_runs/amazon_books_lightgcn_v8_controlled_randomization
run ctrlrand_ml1m ml1m_lightgcn_v8_controlled_randomization \
  python3 scripts/run_controlled_randomization.py --dataset ml1m --source-run $V3M \
  --out results/journal_runs/ml1m_lightgcn_v8_controlled_randomization

# 3. sequential baselines
run seqbaselines_amazon amazon_books_lightgcn_v8_sequential_baselines \
  python3 scripts/run_sequential_baselines.py --dataset amazon --source-run $V3A \
  --out results/journal_runs/amazon_books_lightgcn_v8_sequential_baselines --seeds 42 43 44
run seqbaselines_ml1m ml1m_lightgcn_v8_sequential_baselines \
  python3 scripts/run_sequential_baselines.py --dataset ml1m --source-run $V3M \
  --out results/journal_runs/ml1m_lightgcn_v8_sequential_baselines --seeds 42 43

# 4. matched lambda sweeps
if [ ! -d results/journal_runs/amazon_books_lightgcn_v8_matched_lambda_sweep ]; then
  run matchedsweep_amazon amazon_books_lightgcn_v8_matched_lambda_sweep \
    env C1_WITH_SHAPLEY=1 python3 scripts/run_matched_lambda_sweep.py --dataset amazon \
    --source-run $V3A --out results/journal_runs/amazon_books_lightgcn_v8_matched_lambda_sweep --seeds 42 43 44 45 46
fi

# 5. nested leak-free lambda tuning
run nested_amazon amazon_books_lightgcn_v8_nested_tuning \
  python3 scripts/run_nested_lambda_tuning.py --dataset amazon --source-run $V3A \
  --out results/journal_runs/amazon_books_lightgcn_v8_nested_tuning --seeds 42 43 44
run nested_ml1m ml1m_lightgcn_v8_nested_tuning \
  python3 scripts/run_nested_lambda_tuning.py --dataset ml1m --source-run $V3M \
  --out results/journal_runs/ml1m_lightgcn_v8_nested_tuning --seeds 42 43

# 6. selection x valuation factorial
run factorial_amazon amazon_books_lightgcn_v8_selection_factorial \
  python3 scripts/run_selection_factorial.py --dataset amazon --source-run $V3A \
  --out results/journal_runs/amazon_books_lightgcn_v8_selection_factorial --seeds 42 43
run factorial_ml1m ml1m_lightgcn_v8_selection_factorial \
  python3 scripts/run_selection_factorial.py --dataset ml1m --source-run $V3M \
  --out results/journal_runs/ml1m_lightgcn_v8_selection_factorial --seeds 42

# 7. convergence v2
run convergence_amazon amazon_books_lightgcn_v8_convergence_v2 \
  python3 scripts/run_convergence_v2.py --dataset amazon --source-run $V3A \
  --out results/journal_runs/amazon_books_lightgcn_v8_convergence_v2 --max-users 1000
run convergence_ml1m ml1m_lightgcn_v8_convergence_v2 \
  python3 scripts/run_convergence_v2.py --dataset ml1m --source-run $V3M \
  --out results/journal_runs/ml1m_lightgcn_v8_convergence_v2 --max-users 1000

# 8. multi-draw negative-set sensitivity
for OFF in 1000000 2000000; do
  run negset_draw_${OFF}_amazon amazon_books_lightgcn_v6_negset_sensitivity_draw_${OFF} \
    python3 scripts/run_negset_sensitivity.py --dataset amazon --source-run $V3A \
    --out results/journal_runs/amazon_books_lightgcn_v6_negset_sensitivity_draw_${OFF} \
    --sizes 50 100 500 --max-users 1500 --neg-offset $OFF
done

# 9. multi-seed design ablations (incl. native-vs-kernel intervention rows)
run designabl_amazon_43 amazon_books_lightgcn_v3_prospective \
  python3 scripts/run_design_ablations.py --dataset amazon --seed 43
run designabl_amazon_44 amazon_books_lightgcn_v3_prospective \
  python3 scripts/run_design_ablations.py --dataset amazon --seed 44

echo "FULL-QUEUE DONE ($(date))"
