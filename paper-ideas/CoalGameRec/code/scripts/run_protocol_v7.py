#!/usr/bin/env python3
"""Round-9 REQUIRED FIX #1: corrected temporal protocol (v7).

Implements the coherent sequential protocol demanded by review:
  - the calibration/validation positive i_u^+ (event t-1) is available to all
    methods as context (it already was: player selection, coalition values,
    valid-sim/valid-linear);
  - i_u^+ is EXCLUDED from the test candidate catalog for every family
    (candidates = I \\ (H_u_train U {i_u^+}));
  - reranking z-scores are computed over that corrected candidate set only;
  - test target = event t, unchanged.

Everything else is byte-for-byte the frozen C1b matched-controls protocol
(same script internals, hyperparameters, seeds 42-46, k=24, 100 validation
negatives, lambda=0.10, all nine families incl. Shapley with
C1_WITH_SHAPLEY=1), so the only changed factor relative to v4b/v6 is the
candidate/exclusion handling.

Usage:
  COALGAME_DEVICE=mps C1_WITH_SHAPLEY=1 python scripts/run_protocol_v7.py \\
      --dataset amazon --source-run results/journal_runs/amazon_books_lightgcn_v3_prospective \\
      --out results/journal_runs/amazon_books_lightgcn_v7_corrected_protocol
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matched_controls as rmc  # noqa: E402

_orig_evaluate = rmc.evaluate
_orig_rerank_all = rmc.rerank_all
_orig_valid_sim = rmc.valid_sim_scores_all
_orig_valid_linear = rmc.valid_linear_scores_all


def _val_by_user(split) -> dict[int, int]:
    return dict(zip(split.val.user.astype(int), split.val.item.astype(int)))


def evaluate_v7(scores, split, item_vectors, ks=(5, 10, 20), **kw):
    return _orig_evaluate(scores, split, item_vectors, ks=ks, exclude_by_user=_val_by_user(split))


def rerank_all_v7(base_scores, split, item_vectors, family, **kw):
    return _orig_rerank_all(base_scores, split, item_vectors, family,
                            exclude_by_user=_val_by_user(split), **kw)


def valid_sim_v7(base_scores, split, val_by_user, item_embeddings, lambda_attr=0.10, eps=1e-12):
    return _orig_valid_sim(base_scores, split, val_by_user, item_embeddings, lambda_attr, eps,
                           exclude_by_user=_val_by_user(split))


def valid_linear_v7(base_scores, split, val_by_user, item_embeddings, lambda_attr=0.10, eps=1e-12):
    return _orig_valid_linear(base_scores, split, val_by_user, item_embeddings, lambda_attr, eps,
                              exclude_by_user=_val_by_user(split))


def main():
    # monkey-patch the matched-controls pipeline to the corrected protocol
    rmc.evaluate = evaluate_v7
    rmc.rerank_all = rerank_all_v7
    rmc.valid_sim_scores_all = valid_sim_v7
    rmc.valid_linear_scores_all = valid_linear_v7

    rmc.main()

    # overwrite the manifest with v7 provenance (rmc.main wrote the C1 note)
    import argparse  # noqa: F401  (argv already parsed by rmc.main)
    out_dir = None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--out":
            out_dir = Path(argv[i + 1])
    if out_dir is not None and (out_dir / "manifest.json").exists():
        m = json.loads((out_dir / "manifest.json").read_text())
        m["note"] = ("Protocol v7 (corrected temporal protocol): identical to the frozen C1b "
                     "matched-controls protocol except that the per-user calibration positive is "
                     "excluded from the test candidate catalog for every family and reranking "
                     "z-scores are computed over the corrected candidate set. Review round-9 fix #1.")
        m["candidate_exclusion"] = "candidates = I \\ (H_u_train U {i_u^+})"
        (out_dir / "manifest.json").write_text(json.dumps(m, indent=1))
        print("manifest updated for v7 provenance")


if __name__ == "__main__":
    main()
