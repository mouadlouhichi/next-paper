# ActionShap enhanced paper package

This folder is a self-contained paper-and-results package assembled from commit `e49106a8cec34c0523e36686110c69680c22601b` (`push results`).

## Contents

- `paper.tex`: Springer Nature manuscript source.
- `paper.bib`: manuscript bibliography.
- `sn-jnl.cls`, `sn-basic.bst`: Springer Nature LaTeX template files.
- `final/`: generated schema-v2 result assets only:
  - `tables/`: component AIA, gap, null, convergence, outcome, and protocol tables;
  - `figures/`: manuscript figures and robustness/sensitivity figures;
  - `data/`: compressed user-seed metrics;
  - `manifests/`: validation and asset manifests;
  - `RESULTS_SUMMARY.md`: generated interpretation summary.

## Provenance

The package is derived from commit `e49106a8cec34c0523e36686110c69680c22601b`. The raw JSON result archive is intentionally kept outside ordinary paper assets; its release checksum must be supplied alongside this folder when depositing the artifact. The archive reported for the corresponding local run was:

```text
8760078655018bdb620d2ae72d51317c7bd7a4e65e63c923632de3bcb1ed9c5a
```

Do not treat `validation_report.json` alone as evidence of scientific validity: check the raw archive for user counts, dataset hashes, gate outcomes, and source provenance before submission.
