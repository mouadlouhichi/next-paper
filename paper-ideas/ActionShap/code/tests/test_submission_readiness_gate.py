"""The submission gate has to be able to answer both ways.

``code/scripts/check_submission.py`` is what stands between "we fixed it" and "the
reviewer can see it", so a gate that only ever refuses is as useless as one that only
ever passes. These tests build the two cases the review-9 round actually produced ---
a build that matches its source, and a stale build carrying the previous title page ---
plus the byte-identical duplicate PDF that a "rebuild" commit once contributed, and pin
the verdict for each. They need no TeX distribution, only ``pypdf``, and skip cleanly
without it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
SCRIPTS = CODE / "scripts"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gate = _load("actionshap_check_submission", SCRIPTS / "check_submission.py")

SOURCE = r"""\documentclass[manuscript,screen,review,anonymous]{acmart}
\newcommand{\resultmanifeststamp}{22d42b58733e}
\author{Mouad Louhichi}
\author{Redwane Nesmaoui}
\author{Mohamed Lazaar}
\affiliation{\institution{Universite Mohammed V}}
\begin{document}
\maketitle
The fixed denominator ablation is reported in the supplement.
\end{document}
"""


def _fake_pdf(path: Path, lines: list[str]) -> None:
    """A one-page PDF whose text layer is ``lines``, written without any TeX toolchain."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=600, height=16 * len(lines) + 40)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = StreamObject()
    data = b"BT /F1 12 Tf 12 " + str(12 + 14 * (len(lines) - 1)).encode() + b" Td "
    for line in lines:
        data += b"(" + line.encode("latin-1", "replace") + b") Tj 0 -14 Td "
    stream.set_data(data + b"ET")
    page[NameObject("/Contents")] = stream
    writer.add_metadata({"/Producer": "readiness gate test", "/CreationDate": "D:20270101000000Z"})
    writer.write(str(path))
    writer.close()


@pytest.fixture()
def pair(tmp_path: Path):
    tex = tmp_path / "acmmanuscript.tex"
    tex.write_text(SOURCE)
    return tex, tmp_path


def test_matching_build_produces_no_defects(pair):
    """The gate is not a refusal machine: a build that carries the current source passes."""
    tex, tmp_path = pair
    pdf = tmp_path / "acmmanuscript.pdf"
    _fake_pdf(pdf, [
        "Anonymous Author(s)",
        "Result manifest 22d42b58733e recorded in code/results/manifest.json",
        "The fixed denominator ablation is reported in the supplement.",
    ])
    assert gate.pdf_text_defects(pdf, tex, anonymous=True) == []
    assert "Anonymous" in gate.pdf_page_one(pdf)


def test_stale_build_is_flagged_on_every_axis(pair):
    tex, tmp_path = pair
    pdf = tmp_path / "acmmanuscript.pdf"
    _fake_pdf(pdf, [
        "Mouad Louhichi  Redwane Nesmaoui  Mohamed Lazaar",
        "The permanent artifact URL must be inserted here before submission.",
    ])
    defects = gate.pdf_text_defects(pdf, tex, anonymous=True)
    assert len(defects) == 3, defects
    assert any("page 1 still prints" in d for d in defects)
    assert any("result-manifest stamp" in d for d in defects)
    assert any("must be inserted" in d for d in defects)


def test_camera_ready_build_must_name_the_authors(pair):
    """The same rule read backwards: dropping `anonymous` must bring the names back."""
    tex, tmp_path = pair
    tex.write_text(SOURCE.replace(",anonymous", ""))
    anonymous = "anonymous" in gate.class_options(tex)
    assert not anonymous
    hidden = tmp_path / "acmmanuscript.pdf"
    _fake_pdf(hidden, ["Anonymous Author(s)", "22d42b58733e"])
    assert any("omits the authors" in d
               for d in gate.pdf_text_defects(hidden, tex, anonymous=anonymous))
    named = tmp_path / "named.pdf"
    _fake_pdf(named, ["Mouad Louhichi", "22d42b58733e"])
    assert gate.pdf_text_defects(named, tex, anonymous=anonymous) == []


def test_a_name_in_the_review_body_is_a_leak(pair):
    tex, tmp_path = pair
    assert gate.surnames(tex) == ["Lazaar", "Louhichi", "Nesmaoui"]
    assert [n for n in gate.surnames(tex) if n in gate.review_body(tex)] == []
    # The camera-ready branch of the conditional is not a leak; plain body text is.
    tex.write_text(SOURCE.replace("\\maketitle",
                                  "\\maketitle\n\\ifreviewcopy anonymised \\else Mouad Louhichi \\fi"))
    assert [n for n in gate.surnames(tex) if n in gate.review_body(tex)] == []
    tex.write_text(SOURCE.replace("\\maketitle", "\\maketitle\nMouad Louhichi wrote this."))
    assert [n for n in gate.surnames(tex) if n in gate.review_body(tex)] == ["Louhichi"]


def test_stray_duplicate_build_is_found(tmp_path):
    primary = tmp_path / "acmart-primary"
    (primary / "figures").mkdir(parents=True)
    canonical = {"acmmanuscript": primary / "acmmanuscript.pdf",
                 "supplementary": primary / "supplementary.pdf"}
    for path in canonical.values():
        path.write_bytes(b"%PDF-1.5 canonical")
    (primary / "figures" / "aia_components.pdf").write_bytes(b"%PDF-1.5 figure")
    (primary / "acmmanuscript (1).pdf").write_bytes(b"%PDF-1.5 canonical")
    (tmp_path / "acmmanuscript (1).pdf").write_bytes(b"%PDF-1.5 canonical")
    found = gate.stray_copies([primary, tmp_path], canonical)
    assert len(found) == 2, found
    assert all(twin.name == "acmmanuscript.pdf" for _, twin in found)
    assert not any("figures" in str(pdf) for pdf, _ in found)
