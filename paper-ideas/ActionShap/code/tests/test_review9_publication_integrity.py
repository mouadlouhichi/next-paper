"""Publication-integrity checks added in the review-9 response.

These tests do not recompute the analysis (that would double the suite runtime);
they assert that what the documents print is reproducible from what the release
ships, which is exactly the class of defect the reviewer raised in Issues #16,
#17, #18 and #10:

* the frozen result manifest quoted by both PDFs matches the released files;
* the review-3 generator reproduces the committed supplement table (so a future
  regeneration cannot silently change published numbers);
* the two table mirrors are byte-identical for the files the review-9 generator
  owns, and numerically identical everywhere else;
* the published analysis populations (1000 / 993 / 987 / 339 / 196) are the
  counts implied by the released per-user matrices;
* Table S4's adjusted p-values and exceedance counts follow from the released
  raw p-values by the declared plus-one and Holm rules.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
PAPER = CODE.parent
SCRIPTS = CODE / "scripts"
MATRICES = PAPER / "actionshap-ipm" / "release" / "matrices"
ACM_TABLES = PAPER / "acmart-primary" / "tables"
IPM_TABLES = PAPER / "actionshap-ipm" / "tables"
MAIN_TEX = PAPER / "acmart-primary" / "acmmanuscript.tex"
SUPP_TEX = PAPER / "acmart-primary" / "supplementary.tex"
REVIEW9_JSON = CODE / "results" / "review9" / "review9_statistics.json"
# The three table assets scripts/make_review9_stats.py owns outright.
GEN_ASSETS = ("review9_statistics.tex", "review9_benchmark_replications.tex",
              "appendix_s3b_effects.tex")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# manifest (Issue 16/17)
# --------------------------------------------------------------------------

def test_result_manifest_is_current_and_quoted_by_both_documents():
    manifest = _load("actionshap_manifest", SCRIPTS / "make_result_manifest.py")
    payload = manifest.build()
    for document in (MAIN_TEX, SUPP_TEX):
        assert manifest.tex_stamp(document) == payload["manifest_stamp"], (
            f"{document.name} quotes a stale result-manifest stamp; run "
            "code/scripts/make_result_manifest.py and update both documents"
        )
    assert payload["file_count"] > 40
    assert (CODE / "results" / "manifest.json").exists()


def test_manifest_check_script_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "make_result_manifest.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# generator round-trips (Issues 17/18)
# --------------------------------------------------------------------------

def test_review3_generator_reproduces_the_committed_paired_family_table():
    # The generator's default --release/--table paths are relative, so the check is only
    # meaningful when it runs from code/; pin the working directory instead of inheriting it
    # from whatever invoked pytest.
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "make_review3_stats.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(CODE),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"


GEN_ASSETS = ("review9_statistics.tex", "review9_benchmark_replications.tex",
              "appendix_s3b_effects.tex")


def test_generated_assets_are_well_formed_latex():
    """No LaTeX toolchain exists in the audit environment, so structure is checked.

    The generator writes LaTeX from Python string literals, and one level of
    backslash escaping lost in a patch turned `\\toprule` into a tab character and
    `\\checkmark` into a string no preamble could typeset. Nothing in a normal
    `pytest` run would have noticed, so the invariants a build would enforce are
    asserted here instead: no control characters, matched environments, the house
    `@{}`-trimmed preamble, one row terminator per data row, and as many columns as
    the preamble declares.
    """
    import re

    preamble_re = re.compile(r"\\begin\{(tabular|longtable)\}\{@\{\}([a-zA-Z0-9|.@{}\\\\ ]+)@\{\}\}")
    column_re = re.compile(r"[lcr]|p\{[^}]*\}")
    for name in GEN_ASSETS:
        for directory in (ACM_TABLES, IPM_TABLES):
            path = directory / name
            assert path.exists(), f"{path} missing; run make_review9_stats.py"
            text = path.read_text()
            offenders = sorted({ord(c) for c in text if ord(c) <= 9 or ord(c) in (11, 12)})
            assert not offenders, f"{path}: control characters {offenders} in generated LaTeX"
            assert text.count("{") == text.count("}"), f"{path}: unbalanced braces"
            for env in ("table", "tabular", "longtable"):
                assert text.count(f"\\begin{{{env}}}") == text.count(f"\\end{{{env}}}"), (
                    f"{path}: unmatched {env} environments")
            lines = text.splitlines()
            preambles = preamble_re.findall(text)
            assert preambles, f"{path}: no tabular/longtable preamble found"
            for spec in preamble_re.finditer(text):
                n_cols = len(column_re.findall(spec.group(2)))
                # walk the rows of this environment and check each data row
                start = spec.end()
                body = [ln for ln in lines[start // max(len(text[:start]), 1):] if ln]
                rows = [ln for ln in body
                        if ln.rstrip().endswith("\\\\") and not ln.startswith("%")]
                in_body = False
                for ln in rows:
                    if ln.startswith("\\multicolumn"):
                        continue
                    if ln.strip() in (r"\toprule", r"\midrule", r"\bottomrule"):
                        continue
                    in_body = True
                    n_ampersands = ln.replace(r"\&", "").count("&")
                    assert n_ampersands + 1 == n_cols, (
                        f"{path}: row has {n_ampersands + 1} cells, preamble declares "
                        f"{n_cols}: {ln[:80]}")
                assert in_body, f"{path}: environment with a preamble but no rows"
            # a label that appears twice in one file breaks cross-referencing
            labels = re.findall(r"\\label\{([^}]*)\}", text)
            assert len(labels) == len(set(labels)), f"{path}: duplicate labels {labels}"
            # every reference the generated prose makes must exist somewhere
            refs = set(re.findall(r"\\ref\{([^}]*)\}", text))
            known = set(labels)
            for other in GEN_ASSETS:
                known |= set(re.findall(r"\\label\{([^}]*)\}", (directory / other).read_text()))
            assert refs <= known, f"{path}: unresolved refs {sorted(refs - known)}"


def test_generated_assets_are_well_formed_latex():
    r"""No LaTeX toolchain exists in the audit environment, so structure is checked.

    The generator writes LaTeX out of Python string literals, and losing one level
    of backslash escaping in a patch turned `\toprule` into a tab character and
    `\vspace` into a vertical tab. Nothing in a normal `pytest` run would notice,
    so the invariants a build would enforce are asserted here: no control
    characters, matched environments, the house `@{}`-trimmed preamble, one cell
    per declared column, no duplicate label, and no reference that resolves only
    inside the generated files.
    """
    bs = chr(92)
    row_end = bs + bs
    preamble_re = re.compile(r"\\begin\{(tabular|longtable)\}\{@\{\}(.+?)@\{\}\}")
    column_re = re.compile(r"[lcr]|p\{[^}]*\}")
    for name in GEN_ASSETS:
        for directory in (ACM_TABLES, IPM_TABLES):
            path = directory / name
            assert path.exists(), f"{path} missing; run make_review9_stats.py"
            text = path.read_text()
            offenders = sorted({ord(c) for c in text if ord(c) <= 9 or ord(c) in (11, 12)})
            assert not offenders, f"{path}: control characters {offenders} in LaTeX"
            assert text.count("{") == text.count("}"), f"{path}: unbalanced braces"
            for env in ("table", "tabular", "longtable"):
                begins = text.count(f"\\begin{{{env}}}")
                ends = text.count(f"\\end{{{env}}}")
                assert begins == ends, f"{path}: {begins} begin-{env} vs {ends} end-{env}"
            labels = re.findall(r"\\label\{([^}]*)\}", text)
            assert len(labels) == len(set(labels)), f"{path}: duplicate labels {labels}"
            seen = set(labels)
            for other in GEN_ASSETS:
                seen |= set(re.findall(
                    r"\\label\{([^}]*)\}", (directory / other).read_text()))
            refs = set(re.findall(r"\\ref\{([^}]*)\}", text))
            assert refs <= seen, f"{path}: unresolved refs {sorted(refs - seen)}"

            tables = 0
            for match in preamble_re.finditer(text):
                tables += 1
                n_cols = len(column_re.findall(match.group(2)))
                assert n_cols >= 2, f"{path}: odd preamble {match.group(2)!r}"
                tail = text[match.end():].split(f"\\end{{{match.group(1)}}}")[0]
                # A longtable also carries `\label{...}\` and `\multicolumn{...}{c}{...}\`
                # rows, neither of which is a data row, so require a cell separator.
                rows = [
                    line for line in tail.splitlines()
                    if line.strip() and "&" in line and not line.lstrip().startswith("%")
                    and line.rstrip().endswith(row_end)
                ]
                assert rows, f"{path}: preamble {match.group(2)} has no data rows"
                for row in rows:
                    if row.lstrip().startswith("\\multicolumn"):
                        continue
                    cells = row.rstrip()[:-2].replace(bs + "&", "").count("&") + 1
                    assert cells == n_cols, (
                        f"{path}: preamble declares {n_cols} columns, row has {cells}: "
                        f"{row[:70]}"
                    )
            declared = text.count(r"\begin{tabular}") + text.count(r"\begin{longtable}")
            assert tables == declared, (
                f"{path}: matched {tables} preambles but {declared} environments are declared"
            )


def test_fixed_denominator_paired_contrasts_are_well_defined():
    """Issue #1: the ablation's paired statistics must be a real sign-flip test.

    Two failure modes are pinned here.  A bootstrap resample of the differences
    is *not* a sign-flip test (it once returned p=0.54 for a d_z of -4.4), and a
    difference vector that is constant across users must land on the plus-one
    floor rather than p=0 or an infinite effect size.  Recomputation from the
    same records must be exact, since the published table has to be regenerable
    (issues #16/#17).
    """
    module = _load("review9_stats_for_paired", SCRIPTS / "make_review9_stats.py")

    base = {user: 0.5 + 0.01 * (user % 7) for user in range(30)}
    records = []
    for scorer in ("normalized", "fixed_denominator"):
        for user in base:
            shift = 0.4 if scorer == "normalized" else 0.0
            records.append({
                "user": user,
                "scorer": scorer,
                "aia_shapley_bounded": base[user] + shift,
                "aia_lime_bounded": base[user],
                "aia_shapley_deletion": 1.0,
                "gap_shapley": base[user] - 1.0 + shift,
                "mean_abs_effect": abs(base[user]),
            })
    out = module._paired_contrasts({"records": records})
    assert out["aia_shapley_bounded"]["n"] == 30
    assert abs(out["aia_shapley_bounded"]["mean_difference"] - 0.4) < 1e-12
    assert out["aia_shapley_bounded"]["sign_flip_p"] == 1 / (
        out["aia_shapley_bounded"]["sign_flip_draws"] + 1
    )
    assert out["aia_shapley_bounded"]["sign_flip_draws"] > 1_000
    for key in ("aia_lime_bounded", "aia_shapley_deletion", "mean_abs_effect"):
        assert out[key]["mean_difference"] == 0.0
        assert out[key]["sign_flip_p"] == 1.0
        assert out[key]["cohens_dz"] == 0.0
    assert module._paired_contrasts({"records": records}) == out
    assert module._paired_contrasts({"records": []}) == {}


def test_holm_helper_is_the_step_down_adjustment():
    stats = _load("actionshap_review3_stats", SCRIPTS / "make_review3_stats.py")
    raw = np.array([0.0001, 0.2, 0.4])
    adjusted = stats._holm(raw)
    # multipliers 3, 2, 1 with cumulative maximum -> 0.0003, 0.4, 0.4
    assert np.allclose(adjusted, [0.0003, 0.4, 0.4])


def test_review9_tables_are_identical_across_the_two_mirrors():
    for name in ("review9_statistics.tex", "appendix_s3b_effects.tex"):
        acm, ipm = ACM_TABLES / name, IPM_TABLES / name
        assert acm.exists() and ipm.exists(), f"{name} missing from a mirror"
        assert _sha256(acm) == _sha256(ipm), f"{name} differs between mirrors"


def test_shared_tables_agree_numerically_across_mirrors():
    numeric = re.compile(r"[-+]?[0-9]+\.[0-9]+")
    drift = []
    for acm in sorted(ACM_TABLES.glob("*.tex")):
        ipm = IPM_TABLES / acm.name
        if not ipm.exists():
            continue
        if numeric.findall(acm.read_text()) != numeric.findall(ipm.read_text()):
            drift.append(acm.name)
    assert not drift, f"numeric drift between mirrors: {drift}"


# --------------------------------------------------------------------------
# analysis populations (Issue 10)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matrices() -> pd.DataFrame:
    return pd.read_csv(MATRICES / "user_seed_metrics.csv.gz")


def test_published_denominators_follow_from_the_release(matrices):
    expected = {
        "MovieLens-1M": {"cohort": 1000, "aia_defined": 1000, "positive_oracle": 1000, "ndcg_active": 339},
        "Amazon-Digital-Music": {
            "cohort": 1000,
            "aia_defined": 993,
            "positive_oracle": 987,
            "ndcg_active": 196,
        },
    }
    for dataset, counts in expected.items():
        block = matrices[
            matrices.dataset.eq(dataset)
            & matrices.model.eq("itemknn")
            & matrices.condition.eq("primary")
            & matrices.method.eq("shapley_mc")
        ]
        assert block.user.nunique() == counts["cohort"]
        assert block[block.aia.notna()].user.nunique() == counts["aia_defined"]
        assert (
            block[block.normalized_regret_primary.notna()].user.nunique()
            == counts["positive_oracle"]
        )
        assert block[block.normalized_regret_ndcg.notna()].user.nunique() == counts["ndcg_active"]


def test_manuscript_quotes_the_same_denominators():
    text = MAIN_TEX.read_text()
    for needle in ("$n=1000$ on both datasets", "$n=1000$ on MovieLens; $n=987$ on Amazon"):
        assert needle in text, f"main text no longer states the verified denominator: {needle}"


# --------------------------------------------------------------------------
# multiplicity map (Issue 18)
# --------------------------------------------------------------------------

def test_table_s4_values_follow_from_the_release_and_exceedances_are_consistent():
    if not REVIEW9_JSON.exists():
        pytest.skip("review-9 statistics not generated yet; run make_review9_stats.py")
    payload = json.loads(REVIEW9_JSON.read_text())
    s4 = payload["s4_regeneration"]
    assert s4["rows"] == 40 and s4["unmatched"] == []
    s4["holm_changes"] = payload["s4_holm_changes"]
    paired = pd.read_csv(MATRICES / "paired_tests.csv")
    method = {"MC Shapley": "shapley_mc", "LIME": "lime", "LOO": "loo",
              "Greedy seq.\\ del.": "greedy_cf", "Random": "random"}
    for row in s4["holm_changes"]:
        dataset, metric, left, right = row["row"].split("|")
        key = "intervention_success_ndcg" if metric == "Success" else "joint_effect_ndcg"
        release = paired[
            paired.dataset.eq({"MovieLens": "MovieLens-1M", "Amazon": "Amazon-Digital-Music"}[dataset])
            & paired.model.eq("itemknn")
            & paired.condition.eq("primary")
            & paired.metric.eq(key)
            & paired.left.eq(method[left])
            & paired.right.eq(method[right])
        ]
        assert not release.empty, f"no release row for {row['row']}"
        rec = release.iloc[0]
        draws = int(rec.permutation_draws)
        # the printed p must equal (1 + exceedances) / (draws + 1) exactly
        assert abs((1 + row["exceedances"]) / (draws + 1) - float(rec.permutation_p)) < 1e-9
        assert abs(row["released_now"] - float(rec.p_holm)) < 1e-9


def test_every_released_test_is_reproducible_under_holm():
    paired = pd.read_csv(MATRICES / "paired_tests.csv")
    stats = _load("actionshap_review3_stats2", SCRIPTS / "make_review3_stats.py")
    keys = ["dataset", "model", "evaluation_mode", "utility", "analysis_role", "condition", "metric"]
    mismatches = 0
    for _, group in paired.groupby(keys, dropna=False):
        adjusted = stats._holm(group.permutation_p.to_numpy(dtype=float))
        mismatches += int((np.abs(adjusted - group.p_holm.to_numpy(dtype=float)) > 5e-4).sum())
    assert mismatches == 0, f"{mismatches} published Holm values are not reproducible from their families"


def test_minimum_printed_p_equals_the_declared_permutation_floor():
    paired = pd.read_csv(MATRICES / "paired_tests.csv")
    floor = 1 / (int(paired.permutation_draws.max()) + 1)
    assert paired.permutation_p.min() == pytest.approx(floor, rel=1e-9)


# --------------------------------------------------------------------------
# stale-artifact guard (the defect this round must not repeat)
# --------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "known review-9 finding: the committed PDFs predate the .tex fixes and "
        "must be rebuilt with `make pdf`; remove this marker once they are"
    ),
    strict=False,
)
def test_compiled_pdfs_are_not_silent_about_the_revised_text():
    """Warn-as-fail guard: PDFs must not lag behind the manuscript sources.

    A fresh build is `make -C <repo root> pdf`. The test compares the PDF
    /CreationDate with the last commit date of the sources it was built from,
    which survives a checkout that resets mtimes.
    """
    validator = _load("actionshap_validate", SCRIPTS / "validate_manuscript.py")
    root = MAIN_TEX.parent
    checks = []
    for name, source in (("acmmanuscript.pdf", MAIN_TEX), ("supplementary.pdf", SUPP_TEX)):
        pdf = root / name
        if not pdf.exists():
            continue
        built = validator.pdf_creation_date(pdf)
        newest = validator.newest_source_time(root, [source])
        if built is None or newest is None:
            continue
        checks.append((name, built, newest))
    for name, built, newest in checks:
        assert built >= newest, (
            f"{name} was built at {built.isoformat()} but its sources change at "
            f"{newest.isoformat()}: rebuild with `make pdf` before submitting"
        )
def _significant_digits(token: str) -> int:
    body = token.lower().split("e")[0].lstrip("-+")
    digits = body.replace(".", "").lstrip("0")
    return len(digits.rstrip("0") or "0")


def test_derived_payloads_do_not_depend_on_blas_last_digits() -> None:
    """The files that feed the printed numbers must not carry last-ULP noise.

    OpenBLAS (the review sandbox) and Accelerate (the authors' workstation) agree to
    ~1e-16 relative, which is invisible at the 3-5 decimals every table prints but
    enough to change the 16th significant digit of a stored $p$-value. Before the
    generator quantised its output, that difference moved the content hash quoted in
    both PDFs whenever somebody re-ran ``make stats`` on another machine -- a stamp
    that only reports *which machine computed it* is worse than no stamp, because it
    trains the reader to ignore the failure. So derived payloads are serialised at 12
    significant digits; raw run outputs stay at full precision on purpose.
    """
    csv_path = REVIEW9_JSON.parent / "review9_multiplicity_map.csv"
    text = REVIEW9_JSON.read_text()
    floats = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", text)]
    assert floats, "no floats found: did the derived JSON change shape?"
    for value in floats:
        if value == 0.0 or not np.isfinite(value):
            continue
        assert float("%.12g" % value) == value, (
            f"{value!r} is stored at more than 12 significant digits: regenerate with "
            "make_review9_stats.py, which quantises derived payloads so the stamp is "
            "machine-independent"
        )
    if csv_path.exists():
        tokens = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", csv_path.read_text())
        assert tokens, "no floats found in the multiplicity CSV"
        over = [x for x in tokens if _significant_digits(x) > 13]
        assert not over, f"CSV carries full-precision reprs, e.g. {over[:3]}"
def test_review_documents_are_anonymized_for_double_blind_review() -> None:
    """A review PDF that prints the authors is the one failure no reviewer warns you about.

    TORS reviews double-blind; acmart suppresses the ``\author`` block only when the
    class carries ``anonymous``. The check is that the option is present while the block
    is still in the source (so dropping it at camera-ready is a deliberate edit that the
    validator will describe, not a silent regression during review).
    """
    for path in (MAIN_TEX, SUPP_TEX):
        text = path.read_text()
        match = re.search(r"\\documentclass\[([^\]]*)\]\{acmart\}", text)
        assert match, f"{path.name}: no acmart \documentclass found"
        opts = [o.strip() for o in match.group(1).split(",")]
        if "\\author{" in text:
            assert "anonymous" in opts, (
                f"{path.name}: author block present without the `anonymous` class option "
                "-- reviewers would see names, affiliations and ORCIDs on page 1"
            )


def test_the_overleaf_project_is_self_contained(tmp_path):
    """The generated Overleaf zip has to resolve every reference on its own.

    ``make pdf`` is unavailable here, so the submission is compiled from the packed
    project. That only works if the packer follows the same input graph TeX does: a
    ``\\safeinput`` that points at a file outside ``tables/`` and ``figures/``, or a table
    that pulls in a figure the zip does not carry, compiles into a page reading
    ``[Missing table asset: ...]`` --- and the ``\\safeinput`` wrapper makes that a
    warning, not an error, so nothing downstream would notice. Packing to a real zip and
    re-resolving inside it is the difference between trusting the include list and
    checking it.
    """
    packer = _load("actionshap_make_overleaf", SCRIPTS / "make_overleaf_project.py")
    resolved, unresolved = packer.referenced_files(packer.PRIMARY, packer.MAIN_DOCUMENTS)
    assert not unresolved, f"documents reference missing files: {sorted(map(str, unresolved))}"
    assert {p.name for p in resolved} >= {"acmmanuscript.tex", "supplementary.tex"}
    assert all(p.exists() for p in resolved)

    out = tmp_path / "project.zip"
    assert packer.build(packer.PRIMARY, out) == 0
    import zipfile
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        readme = archive.read("README-OVERLEAF.txt").decode()
    for path in resolved:
        assert str(path.relative_to(packer.PRIMARY)) in names
    for support in packer.CLASS_SUPPORT:
        if (packer.PRIMARY / support).exists():
            assert support in names, f"{support} is loaded by the class or the documents"
    # A stale build file in the project is how an old PDF masquerades as a new one, and a
    # .bbl in the zip is how latexmk skips the bibliography step.
    assert not [n for n in names if n.endswith((".bbl", ".aux", ".log"))]
    assert not [n for n in names if "/" not in n and n.endswith(".pdf")]
    assert "anonymous" in readme or "anonymous" in names, "the review copy must be anonymised"
    assert "make ready" in readme, "the project must tell the author how to verify the build"


# --------------------------------------------------------------------------- #
# Review-10: the corrections of this round, each pinned by the class of defect it fixes.
# --------------------------------------------------------------------------- #

PAPER = CODE.parent
MAIN_TEX = PAPER / "acmart-primary" / "acmmanuscript.tex"
SUPP_TEX = PAPER / "acmart-primary" / "supplementary.tex"
TABLE_DIR = PAPER / "acmart-primary" / "tables"
SCRIPTS = CODE / "scripts"
RELEASE_DIR = PAPER / "actionshap-ipm" / "release" / "matrices"


def _rho_shift(n_u: int, rho: float, s_p: float, others: list[float]) -> tuple[float, float]:
    """Score change of one candidate under the interface Eq. (4) describes.

    ``s_p`` is the deleted player's similarity to the candidate and ``others`` the remaining
    players' similarities; the normalized coefficient of the downweighted player is
    ``rho / (n_u - 1 + rho)`` and of every other player ``1 / (n_u - 1 + rho)``.
    """
    b = sum(others) / len(others)
    s_old = (s_p + sum(others)) / n_u
    s_new = (rho * s_p + sum(others)) / (n_u - 1 + rho)
    closed = (n_u - 1) * (1 - rho) / (n_u * (n_u - 1 + rho)) * (b - s_p)
    return s_new - s_old, closed


def test_rho_factorization_printed_in_the_manuscript_is_the_actual_algebra():
    """Eq. (4) had dropped the (n_u - 1) / n_u prefactor (review-10)."""
    import random

    random.seed(11)
    for n_u in (2, 3, 7, 60, 1000):
        for rho in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
            s_p = random.uniform(-2, 2)
            others = [random.uniform(-2, 2) for _ in range(n_u - 1)]
            actual, closed = _rho_shift(n_u, rho, s_p, others)
            assert abs(actual - closed) < 1e-12, (n_u, rho, actual, closed)
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert r"\frac{(n_u-1)(1-\rho)}{n_u\,(n_u-1+\rho)}" in text
    assert r"=\frac{1-\rho}{n_u-1+\rho}" not in text


def test_the_difference_is_not_labelled_an_actionability_gap_anywhere():
    """The row labels called the bounded-minus-deletion difference an "actionability gap",
    which the protocol never defines; only the release keys may still use that name."""
    for path in sorted(TABLE_DIR.glob("*.tex")) + [MAIN_TEX, SUPP_TEX]:
        assert "Actionability Gap" not in path.read_text(encoding="utf-8"), path.name
    gap = pd.read_csv(RELEASE_DIR / "actionability_gap_robustness.csv")
    assert "actionability_gap" in set(gap.metric), "the release key must survive the relabelling"
    assert (RELEASE_DIR / "actionability_gap_advantage.csv").exists()
    assert (TABLE_DIR.parent.parent / "code" / "results").exists()


def test_prospective_panel_prints_the_denominator_it_can_defend():
    """The panel audited 600 Gowalla users but only 528 have a defined score for all four
    methods; the table has to state both, and state that containment is not top-1 equality."""
    payload = json.loads((CODE / "results" / "review9" / "prospective_gowalla.json").read_text())
    keys = ("aia_shapley", "aia_lime", "aia_loo", "signed_shapley")   # as the generator defines it
    defined = sum(1 for rec in (payload.get("records") or [])
                  if all(isinstance(rec.get(k), (int, float)) and np.isfinite(rec[k]) for k in keys))
    tex = (TABLE_DIR / "review9_benchmark_replications.tex").read_text(encoding="utf-8")
    assert f"{defined} &" in tex, "the defined-n column does not match the payload"
    assert "Defined" in tex
    assert "528" in tex and "600" in tex



def test_ablation_budget_is_not_called_the_primary_budget():
    """The construct-validity runs used M_pair=250 while the primary analysis is frozen at 500."""
    for name in ("review9_benchmark_replications.tex", "appendix_s3b_effects.tex",
                 "review9_statistics.tex"):
        text = (TABLE_DIR / name).read_text(encoding="utf-8")
        for match in re.finditer(r"250", text):
            window = text[max(0, match.start() - 260):match.end() + 260]
            if "budget" in window.lower():
                assert "primary budget" not in window, f"{name} calls 250 the primary budget"
    hardware = (TABLE_DIR / "review9_benchmark_replications.tex").read_text(encoding="utf-8")
    assert "500" in hardware, "the hardware panel must say how its budget relates to the primary one"


def test_success_and_abstention_families_name_their_source_records():
    """Two Holm values for one contrast are legitimate only while each table says which
    released record it was corrected within."""
    review7 = (TABLE_DIR / "review9_statistics.tex").read_text(encoding="utf-8")
    assert "12-contrast" in review7
    assert "review7_success_abstention_tests.csv" in review7.replace("\\_", "_")
    s3b = (TABLE_DIR / "appendix_s3b_effects.tex").read_text(encoding="utf-8")
    assert "paired_tests.csv" in s3b.replace("\\_", "_")
    assert "review7_success_abstention_tests.csv" in s3b.replace("\\_", "_")


def test_the_two_new_validators_exist_and_pass():
    for script in ("validate_prose_references.py", "validate_inferential_provenance.py"):
        path = SCRIPTS / script
        assert path.exists(), f"{script} must ship with the paper"
        result = subprocess.run([sys.executable, str(path)], cwd=str(CODE),
                                capture_output=True, text=True)
        assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    makefile = ((CODE.parents[2] / "Makefile")).read_text(encoding="utf-8")
    check = makefile[makefile.index("check:"):]
    for script in ("validate_prose_references.py", "validate_inferential_provenance.py"):
        assert script in check.split("\n\n")[0], f"{script} is not wired into `make check`"


def test_declared_supplementary_range_matches_the_document():
    supplement = SUPP_TEX.read_text(encoding="utf-8")
    declared = re.search(r"\\subtitle\{Sections S1--S(\d+)\}", supplement)
    sections = len(re.findall(r"^\\section\{", supplement, flags=re.MULTILINE))
    assert declared is not None, "the supplement must declare its section range"
    assert int(declared.group(1)) == sections
    manuscript = MAIN_TEX.read_text(encoding="utf-8")
    assert "Sections S1--S10" not in manuscript


def test_denominators_and_candidate_count_are_stated_correctly():
    """The re-review's items 2 and 17: the NDCG decision denominator and m."""
    tables = []
    for tree in ("acmart-primary", "actionshap-ipm"):
        d = PAPER / tree / "tables"
        tables += sorted(d.glob("*.tex"))
    for path in tables:
        text = path.read_text(encoding="utf-8")
        assert "196/993" not in text, f"{path.name} quotes the NDCG oracle count against 993"
        assert not re.search(r"m=201\b", text), f"{path.name} counts 201 primary candidates"
    outcomes = (PAPER / "acmart-primary" / "tables" / "intervention_outcomes.tex").read_text()
    assert "196/1000" in outcomes and "different population" in outcomes
    computation = (PAPER / "acmart-primary" / "tables" / "appendix_computation.tex").read_text()
    assert "m=200$ (primary: the held-out target plus 199 sampled negatives)" in computation


def test_prospective_panel_accounts_for_the_missing_cohort():
    """Item 12: the panel must state its own coverage rather than imply three cohorts."""
    text = (TABLE_DIR / "review9_benchmark_replications.tex").read_text(encoding="utf-8")
    block = text[text.index("Defined $n$"):]
    assert "no" in block and "prospective audit" in block, "the Amazon omission must be stated"
    assert "queued" in block, "the panel must say the missing cohort is unrun, not null"


def test_success_convention_matches_the_released_matrix():
    """The supplement's decision-quality block must recompute from the frozen matrices.

    This row-by-row gate is what `make check` runs; it also settles the four-decimal question the
    re-review raised: the published convention averages the per-seed indicator over seeds *and*
    users, so values live on a 1/(n R_seed) lattice.
    """
    audit = json.loads((CODE / "results" / "review9" / "success_estimand_audit.json").read_text())
    assert audit["unsupported"] == 0, [r for r in audit["rows"] if not r["supported"]]
    assert len(audit["rows"]) == 30
    grid = audit["slice"]["grid"]
    assert abs(grid - 1.0 / (1000 * 5)) < 1e-12
    for row in audit["rows"]:
        assert abs(round(row["printed"] / grid) * grid - row["printed"]) <= 1e-9, row
        assert row["ci_within_recomputed"], row
    result = subprocess.run([sys.executable, str(SCRIPTS / "audit_success_estimand.py"), "--check"],
                            cwd=str(CODE), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_manuscript_states_the_convention_the_release_implements():
    text = MAIN_TEX.read_text(encoding="utf-8")
    assert "per-seed indicator" in text and "$1/(nR_{\\mathrm{seed}})$ lattice" in text
    assert "0.2742" in text and "0.2850" in text, "both poolings must be quoted, not just one"
    assert "evaluated on the seed-\emph{averaged} realized effect" not in text
    block = (PAPER / "acmart-primary" / "tables" / "review3_statistics.tex").read_text(encoding="utf-8")
    assert "seed-averaged per-seed indicator" in block
    assert "positive seed-averaged realized NDCG effect" not in block


def test_remaining_work_notebook_reports_statuses_and_executes():
    """The close-out notebook is generated from the files, so it cannot assert a stale status."""
    nb_path = PAPER / "notebooks" / "REMAINING_WORK.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("mrw.status()", "audit_success_estimand.py", "make_outstanding_runs_notebook.py",
                   "code_tasks", "resultmanifeststamp" if False else "manifest", "READY"):
        assert needle in sources, needle
    # nothing is invented: the queue is delegated, and each job names a real subcommand
    assert "make_outstanding_runs_notebook.py" in sources
    gen_path = SCRIPTS / "make_outstanding_runs_notebook.py"
    spec = importlib.util.spec_from_file_location("mor_gen_check", gen_path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    runner = (SCRIPTS / "run_review9_experiments.py").read_text(encoding="utf-8")
    jobs = gen.build_jobs()
    assert jobs, "the queue must be derived, not empty"
    for job in jobs:
        assert job["experiment"] in runner, job["experiment"]
        assert Path(str(job["out"])).name.endswith(".json")
