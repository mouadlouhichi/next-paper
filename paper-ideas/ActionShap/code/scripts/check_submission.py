#!/usr/bin/env python3
"""Submission readiness gate: ``make ready``.

Answers one question from the files, so it does not depend on anyone remembering what
was fixed in which round: *can this go to the editor right now?*  Review-9 caught two
classes of defect that every source-level check in this repository passed --- the
compiled PDFs lagged behind the ``.tex`` sources, and a "rebuild" commit carried a PDF
whose bytes were the old build --- so the checks that matter here are made on the
artefacts a reviewer will read, not only on the sources that produced them.

Three groups, in the order a submission falls apart:

* sources      --- anonymisation switches, leftover TODO brackets, and everything
                   ``validate_manuscript.py`` already enforces (run, not duplicated).
* compiled PDFs --- present at the canonical paths, unique, newer than their sources,
                   and actually carrying the current text (anonymised title page, the
                   frozen manifest stamp, no placeholder sentence, the review-9 panels).
* deposit      --- the archive, its checksum sidecar, its per-member manifest, and the
                   claim that it holds the tables the documents typeset.

A BLOCKER is something an editor or reviewer would see in what is being submitted. A
NOTE is expected at this stage (a DOI that cannot exist before the deposit is minted).
Exit status 0 means no blockers. ``pypdf`` is optional: without it the PDF text checks
are reported as unverified instead of passed, because a silent skip is how the stale
PDFs survived a whole review round.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent                                  # paper-ideas/ActionShap
PRIMARY = ROOT / "acmart-primary"
DOCS = {
    "acmmanuscript": PRIMARY / "acmmanuscript.tex",
    "supplementary": PRIMARY / "supplementary.tex",
}
LETTERS = [PRIMARY / "cover_letter.tex", PRIMARY / "cover_letter.md"]

sys.path.insert(0, str(HERE))
import validate_manuscript as V  # noqa: E402  (timestamp helpers, single definition)

CLASSOPT_RE = re.compile(r"\\documentclass\s*\[([^\]]*)\]\{acmart\}")
STAMP_RE = re.compile(r"\\newcommand\{\\resultmanifeststamp\}\{([0-9a-fA-F]{8,})\}")
AUTHOR_RE = re.compile(r"\\author\{([^}]*)\}")
# A bracketed placeholder a human still has to fill in. `\\documentclass[options]`,
# optional arguments and citations are not placeholders, so only brackets holding
# imperative or unknown-text markers count.
TODO_RE = re.compile(
    r"\[[^\]\n]{0,120}\b(?:insert|fill in|to be (?:added|confirmed)|TBD|FIXME|TODO|URL here)\b[^\]\n]{0,120}\]",
    re.IGNORECASE,
)
OLD_TODO_SENTENCE_RE = re.compile(r"must be inserted", re.IGNORECASE)
DEPOSIT_DOI_OK_RE = re.compile(r"\[\s*(DOI|URL)[^\]]*\]", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def class_options(tex: Path) -> list[str]:
    text = tex.read_text(encoding="utf-8")
    match = CLASSOPT_RE.search(text)
    return [o.strip() for o in match.group(1).split(",")] if match else []


def surnames(tex: Path) -> list[str]:
    """Last token of each ``\\author{...}`` block: what must not appear in a review body."""
    out = []
    for name in AUTHOR_RE.findall(tex.read_text(encoding="utf-8")):
        clean = re.sub(r"\\[a-zA-Z]+|[{}]", "", name).strip()
        if clean:
            out.append(clean.split()[-1])
    return sorted(set(out))


def review_body(tex: Path) -> str:
    """What a reviewer reads: from ``\\maketitle`` on, with review/camera-ready
    conditionals unfolded to the branch the review copy takes."""
    text = tex.read_text(encoding="utf-8")
    body = text.rsplit("\\maketitle", 1)[-1]
    return re.sub(r"\\ifreviewcopy(.*?)\\else.*?\\fi", r"\1", body, flags=re.S)


def pdf_page_one(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    return (PdfReader(str(pdf)).pages[0].extract_text() or "")


def pdf_text(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    reader = PdfReader(str(pdf))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def pdf_text_defects(pdf: Path, tex: Path, anonymous: bool) -> list[str]:
    """Why a compiled build does not match this source, as blocker strings.

    These are content checks on purpose. A /CreationDate only proves what the writer
    believed; the review round that shipped a "rebuild" commit carrying the previous
    bytes would have passed any date comparison, so the build has to carry something
    only the current source contains (the frozen manifest stamp), must not carry the
    sentence the current source deleted, and must show the anonymity the source asks
    for. Returns no findings when ``pypdf`` is unavailable rather than pretending to
    pass: the caller reports that as an unverified note.
    """
    if importlib.util.find_spec("pypdf") is None or not pdf.exists():
        return []
    text = pdf_text(pdf)
    page_one = pdf_page_one(pdf) or ""
    listed = surnames(tex)
    stamp = STAMP_RE.search(tex.read_text(encoding="utf-8"))
    defects: list[str] = []
    if anonymous:
        shown = [name for name in listed if name in page_one]
        if shown:
            defects.append(f"page 1 still prints {shown}: the anonymisation in the "
                           "source never reached the build")
        elif "Anonymous" not in page_one:
            defects.append("page 1 shows neither the authors nor an anonymous "
                           "placeholder: the title page is not what the source describes")
    elif listed and not any(name in page_one for name in listed):
        defects.append("this is a camera-ready build (no `anonymous`) whose page 1 "
                       "omits the authors")
    if stamp and stamp.group(1) not in text:
        defects.append(f"does not contain the result-manifest stamp {stamp.group(1)} "
                       "the source quotes: it was built from an older tree, whatever "
                       "its date says")
    if OLD_TODO_SENTENCE_RE.search(text):
        defects.append('still prints the placeholder sentence the current source deleted '
                       '("... must be inserted ... before submission"): an editor would '
                       'read an unfinished manuscript')
    if pdf.stem == "supplementary" and "fixed denominator" not in text.lower():
        defects.append("has no fixed-denominator panel: the review-9 ablation is in the "
                       "source but not in the build")
    return defects


def stray_copies(search_dirs: list[Path], canonical: dict[str, Path]) -> list[tuple[Path, Path]]:
    """Compiled copies that are not the canonical file for their document.

    ``figures/`` is not searched, so vector assets are never mistaken for a build; a
    name like ``acmmanuscript (1).pdf`` is, because that is how a re-download from an
    editor reports a build that was never recompiled.
    """
    found: list[tuple[Path, Path]] = []
    for directory in search_dirs:
        for pdf in sorted(directory.glob("*.pdf")):
            twin = next((target for name, target in canonical.items()
                         if pdf.stem == name or pdf.stem.startswith(name + " ")), None)
            if twin is not None and pdf != twin:
                found.append((pdf, twin))
    return found


def run(*cmd: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


class Gate:
    def __init__(self) -> None:
        self.blockers: list[str] = []
        self.notes: list[str] = []
        self.lines: list[tuple[str, str]] = []

    def ok(self, group: str, message: str) -> None:
        self.lines.append(("  ok   ", f"[{group}] {message}"))

    def note(self, group: str, message: str) -> None:
        self.lines.append(("  note ", f"[{group}] {message}"))
        self.notes.append(message)

    def block(self, group: str, message: str) -> None:
        self.lines.append(("  BLOCK", f"[{group}] {message}"))
        self.blockers.append(message)

    def verdict(self) -> int:
        width = max((len(m) for _, m in self.lines), default=0)
        for mark, message in self.lines:
            print(f"{mark} {message}")
        print()
        if self.blockers:
            print(f"NOT READY TO SUBMIT -- {len(self.blockers)} blocker(s):")
            for item in self.blockers:
                print(f"  - {item}")
            print("\nRe-run `make ready` after fixing; a green run is checked against "
                  "the bytes in this repository, not against the last conversation.")
            return 1
        print("READY TO SUBMIT -- no blockers. "
              f"{len(self.notes)} note(s) accepted at this stage.")
        for item in self.notes:
            print(f"  - {item}")
        return 0


def check_sources(gate: Gate) -> dict[str, list[str]]:
    group = "sources"
    options = {name: class_options(tex) for name, tex in DOCS.items()}

    # The twin of every source-level rule, taken from the validator rather than restated.
    proc = run(sys.executable, str(HERE / "validate_manuscript.py"), cwd=CODE)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        gate.block(group, "validate_manuscript.py did not emit JSON; the source checks "
                          "are unverifiable, which is itself a blocker")
        report = {"status": "FAIL", "errors": [], "warnings": []}
    for error in report.get("errors", []):
        gate.block(group, f"manuscript validation error: {error}")
    if not report.get("errors"):
        gate.ok(group, f"validate_manuscript.py: {report['status']}")
    for warning in report.get("warnings", []):
        if "was built at" in warning or "neither document inputs" in warning:
            continue  # the first is re-checked on the artefacts below; the second is benign
        gate.note(group, f"validator warning: {warning}")

    for name, tex in DOCS.items():
        opts = options[name]
        anonymous = "anonymous" in opts
        if "review" in opts and not anonymous:
            gate.block(group, f"{name}.tex is built with `review` but not `anonymous`: "
                             "the title page prints the authors and a double-blind venue "
                             "returns it unread")
        elif anonymous:
            gate.ok(group, f"{name}.tex declares `anonymous`")
        leaked = [s for s in surnames(tex) if s in review_body(tex)]
        if leaked:
            gate.block(group, f"{name}.tex body names {leaked} even though `anonymous` is "
                              "set (acmart hides only the \\author block; wrap the named "
                              "text in \\ifreviewcopy ... \\else ... \\fi)")
        else:
            gate.ok(group, f"{name}.tex body carries no author surnames")
        for found in TODO_RE.finditer(tex.read_text(encoding="utf-8")):
            gate.block(group, f"{name}.tex still carries a placeholder bracket: {found.group(0)}")
    if not any("placeholder bracket" in b for b in gate.blockers):
        gate.ok(group, "no unfilled placeholder brackets in either document")

    for letter in LETTERS:
        if not letter.exists():
            continue
        text = letter.read_text(encoding="utf-8")
        for found in TODO_RE.finditer(text):
            found = found.group(0)
            if DEPOSIT_DOI_OK_RE.search(found):
                gate.note(group, f"{letter.name}: the deposit DOI/URL slot is open, which "
                                 "it must be until the deposit is minted")
            else:
                gate.block(group, f"{letter.name} has an unanswered placeholder: [{found}]")
    return options


def check_pdfs(gate: Gate, options: dict[str, list[str]]) -> None:
    group = "compiled PDFs"
    canonical = {name: PRIMARY / f"{name}.pdf" for name in DOCS}

    # A gate that skips its central check must not report success, so a missing text
    # extractor is a blocker rather than a footnote: `make check` can stay green for a
    # whole round on top of PDFs nobody read.
    if importlib.util.find_spec("pypdf") is None:
        gate.block(group, "pypdf is not installed, so the compiled text cannot be read; "
                          "run `pip install pypdf` and re-run (this is deliberately a "
                          "blocker, not a skip)")

    for name, pdf in canonical.items():
        if not pdf.exists():
            gate.block(group, f"{pdf.relative_to(ROOT)} does not exist: run `make pdf`")
            continue
        if not pdf_text(pdf) and importlib.util.find_spec("pypdf") is not None:
            gate.block(group, f"{name}.pdf has no extractable text; it is not a build of "
                              "this manuscript")

    # Exactly one compiled copy of each document. A review round ended with a second,
    # byte-identical `acmmanuscript (1).pdf` at the package root: it proved nothing about
    # a rebuild and it is the kind of file that gets uploaded by mistake. Scoped to the
    # two directories that can hold a build, so figure PDFs are not swept up.
    for pdf, twin in stray_copies([PRIMARY, ROOT], canonical):
        detail = ""
        if twin.exists() and sha256(pdf) == sha256(twin):
            detail = " and it is byte-identical to it, so it is not evidence of a rebuild"
        gate.block(group, f"a second copy of {twin.name} sits at "
                          f"{pdf.relative_to(ROOT)}{detail}; a submission carries exactly "
                          "one compiled document per file, at the canonical path")

    for name, tex in DOCS.items():
        pdf = canonical[name]
        if not pdf.exists():
            continue
        built = V.pdf_creation_date(pdf)
        newest = V.newest_source_time(PRIMARY, [tex])
        if built is None:
            gate.block(group, f"{name}.pdf carries no /CreationDate; cannot prove it is current")
        elif newest is not None and built < newest:
            gate.block(group, f"{name}.pdf was built at {built:%Y-%m-%d %H:%M} UTC but "
                              f"{tex.name} last changed at {newest:%Y-%m-%d %H:%M} UTC: "
                              "the reviewer would read the previous version")
        else:
            gate.ok(group, f"{name}.pdf is newer than {tex.name}")
        defects = pdf_text_defects(pdf, tex, "anonymous" in options.get(name, []))
        for defect in defects:
            gate.block(group, f"{name}.pdf: {defect}")
        if importlib.util.find_spec("pypdf") is None:
            gate.note(group, f"{name}.pdf text not read (see the pypdf blocker above)")
        elif not defects and pdf.exists():
            gate.ok(group, f"{name}.pdf text matches its source")


def check_deposit(gate: Gate) -> None:
    group = "deposit"
    archive = CODE / "results" / "release" / "actionshap-schema-v2-results.tar.gz"
    sidecar = Path(f"{archive}.sha256")
    if not archive.exists():
        gate.block(group, f"{archive.relative_to(ROOT)} does not exist: run `make artifact`")
        return
    if not sidecar.exists():
        gate.block(group, "the archive has no .sha256 sidecar, so the checksum quoted in "
                          "the cover letter cannot be verified")
    else:
        recorded = sidecar.read_text(encoding="utf-8").split()[0]
        actual = sha256(archive)
        if recorded != actual:
            gate.block(group, f"the .sha256 sidecar says {recorded[:12]}... but the archive "
                              f"hashes to {actual[:12]}...: the cover letter would point at "
                              "bytes that do not exist")
        else:
            gate.ok(group, f"archive sha256 matches its sidecar ({actual[:12]}...)")

    with tarfile.open(archive) as tar:
        members = {m.name for m in tar.getmembers() if m.isfile()}
    with tarfile.open(archive) as tar:
        # Read the bytes now: a tarfile handle is dead once the archive is closed, and
        # the member list is needed by three checks below.
        blobs = {}
        for member in tar.getmembers():
            if member.isfile():
                stream = tar.extractfile(member)
                if stream is not None:
                    blobs[member.name] = stream.read()
    if "release_checksums.json" not in members:
        gate.block(group, "the archive carries no release_checksums.json, so a reviewer "
                          "cannot check the deposit without re-hashing the whole tree")
    else:
        # The manifest records the raw matrices by basename under `files` and the
        # generated half by arcname under `derived`; the documents claim it
        # "content-addresses each member", so each recorded hash is recomputed here.
        checksums = json.loads(blobs["release_checksums.json"].decode())
        recorded = {f"raw/{e['path']}": e["sha256"] for e in checksums.get("files", [])}
        recorded.update({e["path"]: e["sha256"] for e in checksums.get("derived", [])})
        missing = sorted(members - set(recorded) - {"release_checksums.json"})
        wrong = []
        for arcname, expected in recorded.items():
            blob = blobs.get(arcname)
            if blob is None:
                wrong.append(f"{arcname} (listed but absent from the archive)")
            elif hashlib.sha256(blob).hexdigest() != expected:
                wrong.append(f"{arcname} (hash disagrees)")
        if missing:
            gate.block(group, f"{len(missing)} archived file(s) are not covered by "
                              f"release_checksums.json: {missing[:3]}")
        elif wrong:
            gate.block(group, f"release_checksums.json disagrees with the archive bytes for "
                              f"{len(wrong)} file(s): {wrong[:2]}")
        else:
            gate.ok(group, f"release_checksums.json covers and verifies all "
                           f"{len(members)} members")

    inputs = re.findall(r"\\(?:safeinput|input)\{(tables/review9_[a-z0-9_]+\.tex)\}",
                        DOCS["supplementary"].read_text(encoding="utf-8"))
    for required in sorted(set(inputs)):
        if required not in members:
            gate.block(group, f"the documents say the deposit holds the generated tables, "
                              f"but {required} is not in the archive")
    if not any("review9" in m and m.endswith(".json") for m in members):
        gate.block(group, "no review-9 run payloads in the archive, though the cover "
                          "letter promises the per-user experiment outputs")
    if not any(m.endswith("run_review9_experiments.py") for m in members):
        gate.block(group, "the runner is missing from the archive, so no number in the "
                          "supplement is recomputable from the deposit")
    else:
        gate.ok(group, f"deposit holds the runners, tables, payloads and manifest "
                       f"({len(members)} members)")

    stamp = STAMP_RE.search(DOCS["acmmanuscript"].read_text(encoding="utf-8"))
    blob = blobs.get("results/manifest.json")
    if stamp and blob is not None:
        archived = json.loads(blob.decode())
        if archived.get("manifest_stamp") != stamp.group(1):
            gate.block(group, f"the documents quote manifest stamp {stamp.group(1)} but the "
                              f"archived manifest says {archived.get('manifest_stamp')}: the "
                              "deposit is not the one the paper was typeset against")
        else:
            gate.ok(group, "the archived manifest is the stamp the documents quote")


def main() -> int:
    gate = Gate()
    print("submission readiness (make ready)")
    options = check_sources(gate)
    check_pdfs(gate, options)
    check_deposit(gate)
    return gate.verdict()


if __name__ == "__main__":
    raise SystemExit(main())
