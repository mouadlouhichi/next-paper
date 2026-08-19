# Round 6 → v20 addendum (2026-08-19): executed v6 runs integrated

Companion to `review_v19_round6_response.md`. The authors executed notebook cells 3 and 4 on
their Mac (commit 84eeced, MPS, torch 2.3.1) and the results are now integrated in manuscript v20.

## Executed and integrated (v20)

1. **Second backbone: NGCF** (`*_ngcf_v6_second_backbone`, 5 seeds × 2 datasets, Shapley included).
   - Manuscript: new §"Second backbone: NGCF (cross-architecture replication)" + `tab:second_backbone`.
   - Result: the full family ordering replicates on both datasets — LOO > Shapley > valid-linear >
     matched/non-game controls > unreranked (ML-1M NDCG@20: LOO 0.04907±0.00057 vs Shapley
     0.04760±0.00080; Amazon: 0.01177±0.00317 vs 0.01125±0.00300). Absolute NDCG is lower because
     the shared frozen hyperparameters are not NGCF-tuned; documented in the caption.
   - **Pending:** per-user paired permutation inference for this backbone — the per-user .gz files
     were excluded from the commit by the old `**.gz` gitignore rule (now removed; authors to re-push).

2. **LOO λ-sweep** (`*_lightgcn_v6_lambda_sweep`, 5 seeds × 2 datasets).
   - Manuscript: LOO rows added to `tab:ablation_lambda`; `fig:lambda_sensitivity` regenerated with
     the LOO curve (LOO dominates at every λ>0 on both datasets).

3. **Independently validation-tuned λ** (reviewers' "fair per-method tuning" demand).
   - λ selected per family by validation NDCG@20 (full-catalog, training items masked); test
     evaluated once. Manuscript: new `tab:lambda_tuned`.
   - Result: tuning **widens** LOO's lead — ML-1M LOO 0.06211±0.00040 (+30.0% over tuned uniform,
     +28.8% over tuned additive-pref); Amazon 0.03683±0.00120 (+23.9% over tuned uniform).

## Repository hygiene
- `code/.gitignore`: blanket `**.gz` rule removed (was hiding the released per-user inference
  artifacts); explanatory note added.
- New figure script `scripts/plot_round6_lambda.py`; patch scripts `_patch_round6.py`,
  `_patch_round6_v20.py`; table generator `scripts/make_round6_tables.py` (all released).

## Still awaiting execution (notebook cells 5–8)
- Cell 5: multi-seed design ablations (seeds 43+; seed 42 released).
- Cell 6: multi-seed masked-forward faithfulness (seeds 43–46; seed 42 released).
- Cell 7: attribution stability + model-randomization + perturbation sanity.
- Cell 8: validation-negative-set sensitivity (50/100/500).
- Plus: re-push of the NGCF per-user metrics (.gz) for paired inference.
