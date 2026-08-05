# ActionShap reproducibility record

## Public source

- Repository URL: <https://github.com/mouadlouhichi/next-paper> (the current GitHub remote is private; publish or mirror it before submission)
- Canonical manuscript: `paper-ideas/ActionShap/paper-v3/actionshap.tex`
- Canonical implementation: `paper-ideas/ActionShap/code/`
- Release archive: `paper-ideas/ActionShap/code/results/release/actionshap-schema-v2-results.tar.gz`
- Release archive SHA-256: `ac4c7fb1993458b6b41054974ebff215710e7a8b5894c7aa6af828e94b2a5b0f`
- Core revision commit: `ab8fd2acdadfb03ad003867d10a60a2bd74e40aa` (the commit containing the manuscript, implementation, validators, and compact asset rebuild)
- License: `paper-v3/LICENSE` applies to the manuscript/code/non-data assets;
  `CITATION.cff` provides machine-readable citation metadata. The source data
  remain governed by GroupLens/Amazon terms and are not redistributed.

The asset generator records the exact Git commit, source hashes, dependency
versions, host platform, runtime, and peak RSS in each newly generated raw run
and in `final/manifests/asset_manifest.json`. The immutable archive hash above
identifies the tracked schema-v2 result bundle. A DOI is deliberately not
claimed for this checkout: a DOI requires a separate Zenodo or institutional
repository deposit.

## Frozen protocol

- Temporal split: final event test, penultimate event validation; timestamp and
  original-record-index tie breaking.
- Model fitting: complete training histories.
- Player/scoring profile: latest 20 training interactions; older training
  interactions are not added as fixed scoring context. The 50/100 conditions
  are profile-window sensitivities.
- Candidate set: target plus 199 uniform unseen negatives; full-unseen-catalogue
  subset has 250 matched users.
- Models: ItemKNN primary (200 top non-zero cosine neighbours) and a
  64-dimensional leave-one-out BPR-style profile model for robustness.
- Attribution utility: target margin against the top 10 competitors with a
  unit-temperature sigmoid.
- Intervention: simultaneous weight downscaling from 1 to 0.5; 0.25 and
  deletion are sensitivities.
- Budget: exact no-action/singleton/pair oracle at B=2; B=1 and B=3 are
  joint-decision sensitivities only.
- Seeds: model/attribution 42--46; candidate 1729; user 2718; tie 31415.
- Monte Carlo: `permutations=500` means 500 base permutations and 1,000
  evaluated orders because each order is paired with its reverse.
- LIME: 512 masks, Hamming kernel width 0.25, ridge alpha 1.0.
- Inference: 10,000 user bootstrap draws, 10,000 paired plus-one draws, and
  1,000 matched within-seed AIA-null draws.

## Tracked archive runtime

The schema-v2 recommendation JSON records contain runtime seconds for the 80
recommendation runs. From the tracked archive:

- ItemKNN recommendation runs: 60 runs, approximately 29.8--2,851.6 seconds
  per run, mean approximately 183.0 seconds.
- Profile recommendation runs: 20 runs, approximately 26.9--299.6 seconds
  per run, mean approximately 95.0 seconds.

The older convergence JSONs predate host/runtime fields. New convergence runs
now record the same metadata as recommendation runs. The exact CPU, memory,
and peak RSS for a regenerated suite are therefore taken from the generated
provenance rather than inferred from these historical timings.

## Build and audit commands

```bash
cd paper-ideas/ActionShap/code
python scripts/make_paper_assets.py --raw results/raw --out ../paper-v3
python scripts/validate_manuscript.py \
  --paper ../paper-v3/actionshap.tex \
  --bib ../paper-v3/paper.bib \
  --require-final
python scripts/validate_review_contract.py --paper-root ../paper-v3
```

For the PDF, run the four-pass sequence in `paper-v3/README.md`; the BibTeX
pass is required. Complete user-level and paired-test data remain in CSV/JSON
supplementary assets, while the PDF tables are compact publication views.
