"""The IP&M asset set must satisfy Elsevier's packaging rules, checked not asserted."""
import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_ipm_assets.py"
spec = importlib.util.spec_from_file_location("make_ipm_assets", SCRIPT)
ipm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipm)


def test_assets_build_and_pass_their_own_constraints():
    files = ipm.build()
    assert not ipm.check(files)


def test_title_page_carries_no_abstract_and_manuscript_no_identity():
    files = ipm.build()
    assert chr(92) + "begin{abstract}" not in files["title_page.tex"]
    assert "Louhichi" in files["title_page.tex"]
    # the front matter must be identity-free; CRediT further down legitimately names the authors
    for name in ("manuscript_snippet.tex", "abstract.tex"):
        front = files[name].split("%% ---- statements")[0]
        assert chr(92) + "affiliation" not in front, name
        assert "um5.ac.ma" not in front, name
    assert "CRediT" in files["manuscript_snippet.tex"]


def test_generated_files_are_on_disk_and_current():
    out = Path(__file__).resolve().parents[2] / "ipm-submission"
    for name in ("title_page.tex", "abstract.tex", "highlights.txt", "manuscript_snippet.tex",
                 "cover_letter_IPM.md", "README_SUBMISSION.md"):
        assert (out / name).exists(), name
    files = ipm.build()
    for name, body in files.items():
        on_disk = (out / name).read_text(encoding="utf-8").rstrip()
        assert on_disk == body.rstrip(), f"{name} is stale; run code/scripts/make_ipm_assets.py"


def test_ipm_cover_letter_stays_outside_the_latex_archive():
    """Elsevier uploads the letter separately; shipping it inside the source archive exposes identities."""
    import zipfile
    zip_path = Path(__file__).resolve().parents[2] / "actionshap-overleaf.zip"
    names = zipfile.ZipFile(zip_path).namelist()
    assert not any("cover_letter_IPM" in n for n in names), names
    # the ACM archive ships the ACM letter; the IP&M set must not reuse that archive verbatim
    assert (Path(__file__).resolve().parents[2] / "ipm-submission" / "README_SUBMISSION.md").exists()
