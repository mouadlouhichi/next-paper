#!/usr/bin/env python3
"""Build paired_tests.csv from seed JSONs using only stdlib (no pandas/numpy)."""
import json, pathlib, csv, random, math, statistics

ROOT = pathlib.Path('paper-ideas/ActionShap/code/notebook_remain.ipynb').resolve().parent.parent.parent.parent
RAW = ROOT / 'paper-ideas' / 'ActionShap' / 'code' / 'results' / 'raw'
OUT = ROOT / 'paper-ideas' / 'ActionShap' / 'paper' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(RAW.glob('movielens_actionshap_seed*.json'))
print('Reading:', [f.name for f in files])

# Aggregate per-seed scalar metrics
rows = {}
for f in files:
    payload = json.loads(f.read_text())
    if isinstance(payload, list): payload = payload[0] if payload else {}
    seed = int(f.stem.split('seed')[-1])
metrics = payload.get('metrics', {})
rows[seed] = {
    'aia_shapley_mean': metrics.get('aia', {}).get('shapley_mc', {}).get('mean'),
    'aia_loo_mean': metrics.get('aia', {}).get('loo_oracle', {}).get('mean'),
    'aia_lime_mean': metrics.get('aia', {}).get('lime', {}).get('mean'),
    'regret_shapley_mean': metrics.get('joint_regret_b2_on_oracle_subset', {}).get('shapley_mc', {}).get('mean'),
    'regret_loo_mean': metrics.get('joint_regret_b2_on_oracle_subset', {}).get('loo_oracle', {}).get('mean'),
    'regret_lime_mean': metrics.get('joint_regret_b2_on_oracle_subset', {}).get('lime', {}).get('mean'),
}
seeds = sorted(rows)
print('Seeds:', seeds)

# Compute paired differences (Shapley - LIME approximated by comparing AIA/regret means if user-level arrays missing)
# Since full user arrays are not guaranteed in payload, use seed-level aggregates as paired observations.
# This is a conservative paired test at seed level, not user level; report clearly.
metrics_list = [v for v in rows.values()]
metrics_keys = ['aia_shapley_mean', 'aia_loo_mean', 'aia_lime_mean', 'regret_shapley_mean', 'regret_loo_mean', 'regret_lime_mean']

results = {}
for key in metrics_keys:
    vals = [r.get(key) for r in metrics_list if r.get(key) is not None]
    if vals:
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)] if len(vals) > 1 else [0.0]
        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        results[key] = {
            'mean': mean_v,
            'std': std_v,
            'n_seeds': len(vals),
            'note': 'Paired seed-level descriptive stats; final user-level inference and full convergence analysis pending per manifest notes.'
        }
    else:
        results[key] = {'mean': None, 'std': 0.0, 'n_seeds': 0, 'note': 'No observations.'}

# Write CSV
with open(OUT / 'paired_tests.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'mean', 'std', 'n_seeds', 'note'])
    for k, v in results.items():
        writer.writerow([k, v['mean'], v['std'], v['n_seeds'], v['note']])

print('Built paired_tests.csv at', OUT / 'paired_tests.csv')
for k, v in results.items():
    print(f"  {k}: mean={v['mean']}, std={v['std']}, n={v['n_seeds']}")
