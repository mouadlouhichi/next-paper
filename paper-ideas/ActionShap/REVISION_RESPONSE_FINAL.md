# ActionShap — response to the final revision audits

*Scope: every mandatory and editorial item from the two final audits ("Final Revision Audit" and "Final Check of the Latest Version"). Scientific content is unchanged except where a documented rule was made explicit (minimum-seed floor); all headline numbers were regenerated from the schema-v2 archive with the corrected asset builder and still validate with `status: PASS`, 0 errors, 0 warnings.*

## Mandatory fixes (review 2)

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | `Table ??` / absent Table 2 | Table 2 (`tables/recommendation_quality.tex`) is present in the package and now uses `[H]` placement with a `\FloatBarrier` after §6.1, so it cannot drift or vanish; the `\ref{tab:recommendation-quality}` in §6.1 resolves to Table 2. Data match the reviewer's requested rows exactly (ItemKNN Pass 5/5; Profile Boundary 4/5 and 5/5). | `actionshap.tex` §6.1; `tables/recommendation_quality.tex` |
| 2 | Absent null-calibration Table 5 | `tables/aia_permutation_null.tex` rewritten as the requested primary within-user null-calibration table (observed AIA, null mean, null 95th percentile, plus-one $p$, valid $n$ for all five methods on both datasets) and placed with `[H]` inside §6.4, referenced as `Table~\ref{tab:aia-null}`. | `tables/aia_permutation_null.tex`; `actionshap.tex` §6.4 |
| 3 | §6.4 interrupted by §6.3 floats | All §6.2/§6.3 floats now end with `\FloatBarrier`; the null input moved from §6.3 to §6.4; barriers also close §6.1 and §6.4. Source order now matches the reviewer's required sequence exactly. | `actionshap.tex` §6.1--6.4 |
| 4 | Appendix tables before headings | All appendix tables (B1 contract, C1--C2, D1--D3, E1) converted from `[t]` to `[H]` (float package loaded), so each heading precedes its tables; `\clearpage` order kept as prescribed. | `actionshap.tex` appendices; `tables/appendix_*.tex` |
| 5 | Supplementary numbering | Supplementary headings now read S1, S2, S3 (Tables S3--S5), S4 (Table S6), S5 (Table S7); embedded captions renamed to `Table S3/S4/S5...` (S3a--S3c labels removed); `\nopagebreak`/`\Needspace` keep every heading with its table, eliminating blank heading-only pages. | `supplementary.tex`; `tables/appendix_s3*.tex`, `appendix_contract_supp.tex` |
| 6 | Conditional availability statement | Repository will be public (owner action) under release tag `actionshap-v1`; §6.6 and the Code/materials declaration now use completed "publicly available in release [URL/tag]" wording, commit `0673378`, archive SHA-256 `ac4c…b0f`, and the release directory ships `validation_report.json`, `asset_manifest.json`, `config_final.yaml`, lockfile, scripts, and all CSV matrices. | `actionshap.tex` §6.6 + Declarations; `RELEASE_METADATA.md`; `release/README.md` |

## Editorial corrections (review 2)

- **Greedy name:** `METHOD_LABELS` changed to "Greedy sequential deletion"; Figures 2--3 and all per-condition figures regenerated from the raw archive; legends verified visually. |
- **Reference [36]:** samek entry rewritten with `{AI}` protection, `series = {Lecture Notes in Computer Science}`, `volume = {11700}`, no duplicated title concatenation; `[35]` URL removed (arXiv ID already in journal field). Capitalization protected for `{AI}`, `{KernelSHAP}`, `{Shapley}`, `{BERT4Rec}`, `{LightGCN}`. |
- **Table D3:** caption is now "Primary signed diagnostic metrics for the ItemKNN target-margin analysis", explicitly labelled descriptive (confirmatory targets are Table D1), with Precision@3 identified as the compound metric. |
- **Table C2 dashes:** footnote added: "— denotes a value intentionally omitted from this compact summary, not an unavailable result." |

## Review-1 items closed in the same pass (including final sweep)

Additional final-sweep closures: one-sentence statement that the exact Shapley value satisfies efficiency, symmetry, dummy, and additivity (after Eq. 14); LIME solved by closed-form ridge (no iterative initialisation) stated; BPR short-history behaviour stated (users with <2 distinct items contribute no triple; eligibility floor governs the cohort); keyword "Actionability" narrowed to "Bounded actionability"; responsible-AI/oversight paragraph added to the Discussion; Table 3 caption expands abbreviations (Shapley/Greedy/Random); contract-table null draws renamed to $R_{\mathrm{null}}$.

## Review-1 items closed earlier in this revision

- Weighted-profile circularity removed: utilities/effects are functions of $(S,w)$ with $W_u(S,w)$; $z_u(S):=z_u(S,\mathbf1)$ abbreviations; deletion explicitly uses the zero-profile convention (Eqs. 2--3, 9--10).
- Player identity, minimum-history eligibility ($\ge4$ interactions), and duplicate-interaction rule stated (Eq. 1).
- BPR optimizer/initialization/batch/triplet/regularization details added (Eq. 4).
- $\operatorname{sign}(0)=0$, compound Precision@k with zero-player edge, and stability $<2$-pair rule stated.
- Algorithm 4 states the sorting-vs-enumeration equivalence for additive unit-cost benefits; Algorithm 5 uses $R_{\mathrm{null}}$ and states the minimum-valid-seed rule (implemented as `MIN_VALID_SEEDS = 3` in `make_paper_assets.py`; on the released data every retained user has 5/5 valid seeds, 7 Amazon users 0/5, so numbers are unchanged).
- Individual-method NRegret 95% CIs and active-oracle fractions added to Table 4's footnote; pairwise CIs remain in Table S5.
- Figure 4 (convergence) regenerated with a complete shared legend mapping every curve to dataset--model--utility and both selection panels (rank, Jaccard); caption updated.
- Hypotheses H1/H2 and the operational rationale for $\rho=0.5$ added to the Introduction; asymptotic complexity paragraph added to §6.5; novelty/non-overlap paragraph and comparison-criteria sentence added to the Discussion; artifact/selection and statistical-conclusion validity subsubsections added; Table 1 (components) now displays 95% CIs; Fig. 1 caption cross-references Table D3; conclusion says "fully specified, release-backed protocol".

## Verification

- `pytest`: 95 passed, 1 legacy skip (test updated for the ≥3-seed floor).
- Full asset regeneration from `actionshap-schema-v2-results.tar.gz` with the corrected builder: `validation_report.json` = PASS (0 errors, 0 warnings, 34 disclosed notes); Amazon primary still $n=993$/missing 7.
- Structural LaTeX check: no unresolved `\ref`, all `\safeinput` assets present, balanced environments.
- Release directory re-synced with regenerated matrices, manifest, and validation report; figures re-synced to `figures/`.

## Owner action required before submission

1. Flip `mouadlouhichi/next-paper` to **public** and ensure tag `actionshap-v1` exists on the release commit (the session branch will be pushed; tag push requires owner push rights).
2. Recompile main + supplement from `actionshap-overleaf/` (pdfLaTeX → BibTeX → pdfLaTeX ×2) so Tables 2/5 and `[H]` placements render in the PDF.

## Post-review Overleaf fix (root cause of the original compile failure)

The Overleaf log showed `Missing \endgroup` / `Division by 0` / vanished
Tables 2 & 5. Root cause: `sn-jnl` wraps every table in the real
`threeparttable`, whose tabular-measurement hooks are incompatible with
`\resizebox`-wrapped tabular bodies. All `\resizebox` wrappers were removed
(`tables/recommendation_quality.tex`, appendix contract table, legacy
`protocol_audit`/`attribution_stability`, paper-v3 generated tables, and the
`write_tex` generator). Re-verified with a real-`threeparttable` compile:
main 29 pages / 0 errors, supplement 37 pages / 0 errors.

## Table-width overhaul (final local compile: 0 errors, 0 overfull hboxes)

Following the suggested Springer-friendly approach, all overflowing tables were
converted to `tabularx`/`tabular*` with tightened `tabcolsep` and shortened
cell text: Table 2 (`tabularx`, flexible first column, ML-1M/Amazon labels),
Table 3 (X columns for the CI fields), Table 5 (`tabularx`, null means in
10^-4 units), Table 1 and Table 8 (`tabularx` X columns, hyphen-friendly
cell wording), contract/D3/convergence appendix tables (tightened spacing,
short headers). Verified by full local pdfLaTeX compiles with the real
`threeparttable`: main 30 pages / 0 errors / 0 overfull; supplement
37 pages / 0 errors / 0 overfull.
