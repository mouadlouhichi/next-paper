# IP&M submission assets

| file | upload as | note |
|---|---|---|
| `title_page.tex` | Title page (separate file) | **no abstract**, by design |
| `manuscript_snippet.tex` | head + tail of the main file | abstract, keywords, all five statements |
| `abstract.tex` | paste into the manuscript / submission form | 200 words (limit 200) |
| `highlights.txt` | Highlights file | 4 bullets, longest 75 chars (limit 85) |
| `cover_letter_IPM.md` | cover letter | not inside the manuscript zip |
| `../actionshap-overleaf.zip` | LaTeX source | ACM-styled; see TODO below for the elsarticle conversion |

Supplementary material is uploaded as a separate file (37 tables), and the figures are the 4 vector
files already referenced by the manuscript.

## TODO before the final upload (needs the author)
1. Compile the manuscript with `elsarticle` (Overleaf ships it). The body of the ACM file can be
   included as-is; only the front matter and the `\safeinput` table wrapper need the elsarticle
   equivalents.
2. Confirm IP&M's review choice (single- vs double-anonymised). The title page here carries identity and
   the manuscript does not, which satisfies both; if single-anonymised is required, move nothing ---
   upload both files as they are.
3. Fill the ORCID fields in the submission system (not in the LaTeX).
4. Re-check the word count line on the title page after the final edit.
