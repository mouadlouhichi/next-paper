# IP&M submission checklist

Uploaded through the Editorial Manager, in this order:

| # | item | our file | state |
|---|---|---|---|
| 1 | Title page (separate) | `title_page.tex` | no abstract, by design; identities here only |
| 2 | Main file | `manuscript_ipm.tex` + body/table/figure inputs | elsarticle, numbered refs, line numbers |
| 3 | Abstract for the form | `abstract.tex` | 196 words (cap 200) |
| 4 | Keywords for the form | `manuscript_ipm.tex` front matter | 6 (cap 6) |
| 5 | Highlights | `highlights.txt` | 4 bullets, longest 75 chars (cap 85) |
| 6 | Cover letter | `cover_letter_IPM.md` | must NOT be inside the LaTeX archive |
| 7 | Supplementary material | `../acmart-primary/supplementary.tex` | uploaded separately, compiled alone |
| 8 | Figures | `figures/` in the archive | vector, referenced by relative path |
| 9 | Tables | `tables/` in the archive | editable LaTeX, not images |
| 10 | Declarations | in `manuscript_ipm.tex` / `manuscript_snippet.tex` | Data availability, CRediT, competing interest, GenAI, funding, acknowledgements |
| 11 | ORCID iDs | submission form | author-side, not in LaTeX |
| 12 | Review model | submission form | title page + anonymous body satisfies either choice |

Not uploaded (kept in the repository for reproducibility): `../actionshap-overleaf.zip` (ACM build), the
result manifest regeneration logs, the run queue notebooks.

Before submitting: compile `manuscript_ipm.tex` and `supplementary.tex` in Overleaf, confirm no
`Overfull \hbox` in the log, and check that every cross-reference resolves (the ACM `\safeinput`
wrapper is shimmed to \input, so a missing table shows as an error rather than silently vanishing).
