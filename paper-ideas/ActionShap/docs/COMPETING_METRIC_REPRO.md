# Reproducing a published fidelity metric as a competing evaluator

**Why.** Two reviewer-side risks remain after the statistical work: the bibliography is thin on
2023-2026 work, and no *published* evaluation protocol is executed for comparison. Positioning tables
that cite ACCENT, PRINCE, CEERS, RankingSHAP or RankSHAP qualitatively answer "do you know the
literature"; they do not answer "would the field's current metric have reached the same verdict". This
spec closes the second gap with one experiment, not a new model.

**The claim to test.** Standard perturbation faithfulness (deletion/insertion curves, area under them,
or the drop in the ranking metric when top-attributed factors are removed) evaluates an attributor by
how much model output moves when factors are *removed*. ActionShap claims that this measures an
infeasible intervention and therefore misranks attributors relative to a bounded, executable one.

**Design.**
1. Reuse the frozen cohorts and the existing per-user matrices: `condition == "primary"`,
   `model == "itemknn"`, 1,000 users, five seeds, on MovieLens-1M and Amazon-Digital-Music.
2. For each of the five attributors, compute the published metric on the *same* attributions already
   stored, with no re-fitting: sort players by |attribution|, remove the top-$k$ (and separately
   insert from the bottom), record the change in NDCG@10 and in the target-margin utility, and take the
   area/trapezoid over $k = 1..n_u$ as the source papers define it.
3. Rank the attributors by that metric, and independently by bounded AIA and by realized NDCG effect.
4. Report the inversions: which method pairs the published metric ranks one way and the bounded
   protocol ranks the other, with a distinct-user bootstrap CI on the rank difference and a sign-flip
   permutation $p$-value, both at the existing Holm family.

**Acceptance.**
- Everything recomputes from released files; no new model fits, no new seeds.
- The deliverable is one table (attributor x {published metric, bounded AIA, realized effect}) plus one
  sentence in Section 6 naming at least one inversion, or the honest negative result that none exists.
- If the published metric's authors define it for text/tabular features only, state the adaptation and
  keep the adaptation in the code, not in the prose.

**Expected outcome, from what the paper already shows.** Under deletion the ordering favours LOO
(algebraic ceiling of 1.0), and under bounded interventions LOO loses its advantage, so a deletion-based
fidelity metric should invert at least the Shapley-versus-LOO comparison. That is the finding worth
stating, and it is the kind of result a reviewer cannot dismiss as protocol self-favouritism, because it
is produced by the competitor's own definition.

**Do not** cite the reviewer's suggested titles that could not be verified as existing works; three
recent ones were verified and are cited (RankingSHAP, RankSHAP at ICLR 2025, RecourseBench), and the
rest should only enter the bibliography after the same check.
