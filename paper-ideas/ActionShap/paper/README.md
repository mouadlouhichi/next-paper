# ActionShap paper package

This directory is the canonical manuscript package for the recommendation-only ActionShap paper.

## Status

**Manuscript scaffold — empirical results pending.** The sandbox does not contain the user's local MovieLens data and does not have the ActionShap virtual-environment dependencies installed. No numerical result has been fabricated here.

Run `code/ActionShap_All.ipynb` on the machine containing `ratings.dat`. After the run, copy or configure the generated assets into:

```text
paper/figures/
paper/tables/
paper/data/
paper/manifests/
```

The paper must not be submitted until the placeholders in `paper.tex` have been replaced by the generated values and the manifest records the source result files.

## Files

- `paper.tex` — recommendation-only manuscript scaffold.
- `paper.bib` — bibliography for the current framing.
- `figures/`, `tables/`, `images/` — generated-asset destinations; currently empty by design.

The all-in-one execution notebook is:

```text
../code/ActionShap_All.ipynb
```
