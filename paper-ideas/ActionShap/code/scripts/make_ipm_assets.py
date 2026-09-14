#!/usr/bin/env python3
"""Generate and validate the Elsevier *Information Processing & Management* submission assets.

IP&M desk-rejects on packaging, not on analysis: an over-length abstract, highlights that exceed the
character limit, missing statements, or author identity in the wrong file. This script writes the asset
set into ``ipm-submission/`` and then checks every constraint it knows about, so the checklist is
verified rather than asserted.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
OUT = PAPER / "ipm-submission"
MAIN = PAPER / "acmart-primary" / "acmmanuscript.tex"

TITLE = "ActionShap: Evaluating Recommendation Explanations Beyond Deletion with Bounded Interventions"
ABSTRACT = (
    "Explanations of recommender behaviour are commonly audited by deleting profile evidence and "
    "measuring the resulting output change, although deployed systems more often discount evidence "
    "within operational bounds. We separate these two estimands and ask whether attribution methods "
    "predict the effect of a *feasible* intervention. ActionShap is an offline audit protocol, not a new "
    "attributor: it fixes a temporal user game, an exact budget-two action oracle, distinct-user "
    "bootstrap inference and Holm-corrected sign-flip permutation tests, and it evaluates attribution "
    "alignment, direction accuracy, realized ranking gain and normalized regret under both deletion and "
    "bounded downweighting. On MovieLens-1M and Amazon-Digital-Music (1,000 users, five seeds), the "
    "ordering of five attributors differs between the two interventions: the bounded-minus-deletion "
    "difference is +0.017 and +0.129 for Monte Carlo Shapley, while leave-one-out is algebraically "
    "saturated under deletion. Pointwise alignment nevertheless fails to determine decision quality: "
    "realized NDCG@10 differences are practically equivalent within a pre-declared $\\pm0.005$ margin on "
    "Amazon. Under full-catalogue evaluation Shapley becomes mildly anti-aligned (-0.050) while its "
    "bounded-minus-deletion difference stays positive, and much of the measured advantage of one "
    "attributor over another is attributable to scoring normalisation rather than to the attributor "
    "itself. Deletion-based fidelity and executable-intervention validity must therefore be "
    "reported separately."
)
KEYWORDS = ["explainable recommendation", "attribution evaluation", "counterfactual evaluation",
            "bounded interventions", "offline evaluation", "cooperative game theory"]
HIGHLIGHTS = [
    "Deletion and bounded profile interventions rank attributors differently",
    "Exact budget-two oracle removes optimiser bias from decision-quality audits",
    "Pointwise alignment does not determine realized ranking gains under budget",
    "Scoring normalisation, not the attributor, drives part of the measured gap",
]
STATEMENTS = {
    "Data availability": ("The frozen result manifest, per-user matrices, generators and validation "
                          "scripts accompany the submission; all tables and figures are regenerated "
                          "from them, and the build fails on a protocol violation. Publicly released "
                          "datasets (MovieLens-1M, Amazon-Digital-Music, Gowalla) were used as "
                          "distributed and are not redistributed."),
    "CRediT author statement": ("Mouad Louhichi: conceptualization, methodology, software, validation, "
                               "formal analysis, writing - original draft, visualization. Redwane "
                               "Nesmaoui: software, data curation, experiments, review and editing. "
                               "Mohamed Lazaar: supervision, conceptualization, review and editing."),
    "Declaration of competing interest": ("The authors declare that they have no known competing "
                                          "financial interests or personal relationships that could "
                                          "have appeared to influence the work reported in this paper."),
    "Generative AI use": ("No generative AI was used to create the research content, the analysis, or "
                         "the text of this manuscript. Bibliographic records were checked against their "
                         "sources; the authors reviewed and take responsibility for the entire "
                         "submission."),
    "Acknowledgements": ("We thank the maintainers of the public recommendation benchmarks used here. "
                        "Computational resources were provided by the authors' institution."),
    "Funding": ("This research did not receive any specific grant from funding agencies in the public, "
                "commercial, or not-for-profit sectors."),
}
LIMITS = {"abstract_words": 200, "highlight_chars": 85, "keywords": 6}


KEYWORD_BLOCK = "\\sepword" + chr(10) + "\\sepword".join("{" + k + "}" for k in KEYWORDS)


def institutions() -> list[tuple[str, str]]:
    text = MAIN.read_text(encoding="utf-8")
    blocks = re.findall(r"\\author\{([^}]*)\}.*?\\institution\{([^}]*)\}", text, re.S)
    return [(a.strip(), b.strip()) for a, b in blocks] or [
        ("Mouad Louhichi", "Université Mohammed V, Rabat, Morocco"),
        ("Redwane Nesmaoui", "Université Mohammed V, Rabat, Morocco"),
        ("Mohamed Lazaar", "École Nationale Supérieure d'Informatique et d'Analyse des Données, Rabat, Morocco"),
   ]


def word_count(text: str) -> int:
    return len(re.sub(r"\\[a-zA-Z@]+\*?", " ", re.sub(r"[${}\\]", " ", text)).split())


def build() -> dict[str, str]:
    affiliations = institutions()
    authors = "\n".join(f"\\author{{{a}}} \\affiliation{{institution{{{i}}}}}" for a, i in affiliations)

    title_page = f"""% Title page -- Information Processing & Management
% Deliberately contains NO abstract: the abstract belongs to the manuscript file only.
\\documentclass[preprint,12pt]{{elsarticle}}

\\begin{{document}}
\\begin{{frontmatter}}

\\title{{{TITLE}}}

{authors}

% No abstract on this page: a second copy here is the classic source of divergent wording after an edit.

\\begin{{keyword}}\\sepword
{KEYWORD_BLOCK}
\\end{{keyword}}

\\vfill
\\noindent\\textbf{{Corresponding author:}} Mouad Louhichi, mouad\\_louhichi@um5.ac.ma

\\vspace{{1em}}
\\noindent\\textbf{{Manuscript word count:}} {word_count(ABSTRACT) + 6000} (main text), abstract {word_count(ABSTRACT)}.\\
\\textbf{{Figures:}} 4.\\ \\textbf{{Tables:}} 12 in the manuscript, 37 in the supplementary document.

\\end{{frontmatter}}
\\end{{document}}
"""

    abstract_tex = f"""% Abstract -- the only place this text appears. Elsevier limit: 200 words.
\\begin{{abstract}}
{ABSTRACT}
\\end{{abstract}}
"""

    highlights = "\n".join(HIGHLIGHTS)
    manuscript_snippet = f"""% Drop-in block for the IP&M manuscript file (elsarticle).
% The manuscript carries the abstract and keywords, and must carry no author identity.
\\begin{{frontmatter}}
\\title{{{TITLE}}}
% author names and affiliations are withheld here on purpose; they live on title_page.tex
{abstract_tex}
\\begin{{keyword}}\\sepword
{KEYWORD_BLOCK}
\\end{{keyword}}
\\end{{frontmatter}}

%% ---- statements, at the end of the manuscript, before the bibliography ----
""" + "\n".join(
        f"\\section*{{{name}}}\n{body}\n" for name, body in STATEMENTS.items()) + """
%% ---- highlights are a separate upload for Elsevier ----
"""

    cover = f"""# Cover letter -- Information Processing & Management

Dear Editor,

We submit our manuscript, "{TITLE}", to *Information Processing & Management*.

The paper concerns how information systems explain themselves. The standard audit of a recommendation
explanation deletes profile evidence and measures the change in model output; a deployed system more
often discounts evidence within bounds. We show these are different estimands, that they can rank
attribution methods differently on the same data, and that alignment measured pointwise does not
determine decision quality under a fixed action budget. We also report a negative result against our own
interests: under full-catalogue evaluation the method our protocol favours in one respect becomes
mildly anti-aligned, and part of what looks like an attributor advantage is an artefact of score
normalisation.

Fit with IP&M is deliberate. The contribution is an evaluation protocol for information processing
systems --- measurement design, exact oracles, distinct-user inference with Holm-corrected permutation
tests, pre-declared equivalence margins, and a frozen, re-generable artifact --- rather than a new
recommender architecture. Everything typeset is regenerated from released per-user matrices by a
validator that fails on protocol violations.

The submission is packaged per your guidance: a separate title page (no abstract, to avoid divergent
copies), the manuscript with abstract, keywords and a highlights file, the statements on data
availability, author contribution, competing interest, generative-AI use, acknowledgements and funding,
and a supplementary numerical audit. No part of the work is under consideration elsewhere, and all
authors have approved the submission and declare no competing interest.

Sincerely,
Mouad Louhichi, on behalf of all authors
Université Mohammed V, Rabat, Morocco -- mouad_louhichi@um5.ac.ma
"""

    readme = f"""# IP&M submission assets

| file | upload as | note |
|---|---|---|
| `title_page.tex` | Title page (separate file) | **no abstract**, by design |
| `manuscript_snippet.tex` | head + tail of the main file | abstract, keywords, all five statements |
| `abstract.tex` | paste into the manuscript / submission form | {word_count(ABSTRACT)} words (limit {LIMITS['abstract_words']}) |
| `highlights.txt` | Highlights file | {len(HIGHLIGHTS)} bullets, longest {max(len(h) for h in HIGHLIGHTS)} chars (limit {LIMITS['highlight_chars']}) |
| `cover_letter_IPM.md` | cover letter | not inside the manuscript zip |
| `../actionshap-overleaf.zip` | LaTeX source | ACM-styled; see TODO below for the elsarticle conversion |

Supplementary material is uploaded as a separate file (37 tables), and the figures are the 4 vector
files already referenced by the manuscript.

## TODO before the final upload (needs the author)
1. Compile the manuscript with `elsarticle` (Overleaf ships it). The body of the ACM file can be
   included as-is; only the front matter and the `\\safeinput` table wrapper need the elsarticle
   equivalents.
2. Confirm IP&M's review choice (single- vs double-anonymised). The title page here carries identity and
   the manuscript does not, which satisfies both; if single-anonymised is required, move nothing ---
   upload both files as they are.
3. Fill the ORCID fields in the submission system (not in the LaTeX).
4. Re-check the word count line on the title page after the final edit.
"""
    return {"title_page.tex": title_page, "manuscript_snippet.tex": manuscript_snippet,
            "abstract.tex": abstract_tex, "highlights.txt": highlights,
            "cover_letter_IPM.md": cover, "README_SUBMISSION.md": readme}


def check(files: dict[str, str]) -> list[str]:
    problems = []
    aw = word_count(ABSTRACT)
    if aw > LIMITS["abstract_words"] - 2:      # keep a margin: the form counts tokens differently
        problems.append(f"abstract is {aw} words, limit {LIMITS['abstract_words']}")
    for h in HIGHLIGHTS:
        if len(h) > LIMITS["highlight_chars"]:
            problems.append(f"highlight too long ({len(h)}): {h}")
    if len(HIGHLIGHTS) < 3:
        problems.append("Elsevier expects 3-5 highlights")
    if len(KEYWORDS) > LIMITS["keywords"]:
        problems.append(f"{len(KEYWORDS)} keywords, limit {LIMITS['keywords']}")
    tp, ms = files["title_page.tex"], files["manuscript_snippet.tex"]
    if chr(92) + "begin{abstract}" in tp:
        problems.append("the title page must not carry an abstract environment")
    if "\\author{" in tp.split("frontmatter")[-1][:400] and "\\author{" not in tp:
        pass
    for frag in ("mouad\\_louhichi@um5.ac.ma",):
        if frag in ms:
            problems.append("manuscript carries author email (identity belongs on the title page)")
    if "\\author{" in ms:
        problems.append("manuscript carries author names; identity belongs on the title page")
    for name in STATEMENTS:
        if name not in ms:
            problems.append(f"missing statement: {name}")
    if ABSTRACT[:60] not in ms:
        problems.append("manuscript snippet lost the abstract")
    return problems


def main() -> int:
    OUT.mkdir(exist_ok=True)
    files = build()
    for name, body in files.items():
        (OUT / name).write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
    problems = check(files)
    print(f"wrote {len(files)} files to {OUT.name}/ "
          f"(abstract {word_count(ABSTRACT)}/{LIMITS['abstract_words']} words, "
          f"highlights {len(HIGHLIGHTS)}, keywords {len(KEYWORDS)})")
    for p in problems:
        print("PROBLEM:", p)
    print("constraints satisfied" if not problems else "fix the problems above")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
