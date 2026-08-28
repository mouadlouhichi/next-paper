#!/usr/bin/env python3
"""Build the Overleaf project for the review copy: ``make overleaf``.

The submission is compiled on Overleaf because the sandbox has no TeX distribution, and a
zip assembled by hand is how a build ends up disagreeing with the repository: a stale
``.bbl``, a missing table, or --- as happened in review 9 --- a downloaded PDF committed
beside the sources instead of over the artifact that the manifest hash describes. So the
project is generated from the sources rather than curated: the files the documents
actually reference are resolved first, then packed, then re-resolved *inside the zip* to
prove nothing dangles.

Excluded on purpose:

* the compiled PDFs, because shipping them would let an old build look like a new one;
* ``*.bbl``, because the repository's copy predates the current text by ten citations and
  a present-but-stale ``.bbl`` is exactly what lets latexmk skip BibTeX;
* the acmart distribution files (``acmart.dtx``/``.ins``, samples) beyond the class, the
  bibliography style and the two ``.bib`` files the documents load.

The manifest stamp and the source revision are read from the tree and written into
``README-OVERLEAF.txt``, so the project says which state of the repository it represents.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent                                  # paper-ideas/ActionShap
PRIMARY = ROOT / "acmart-primary"
DEFAULT_OUTPUT = CODE / "results" / "release" / "build" / "actionshap-overleaf.zip"

MAIN_DOCUMENTS = ["acmmanuscript.tex", "supplementary.tex", "cover_letter.tex"]
# Loaded by the class or the documents rather than by an input chain, so they cannot be
# discovered by the scan below and have to be named.
CLASS_SUPPORT = ["acmart.cls", "ACM-Reference-Format.bst", "actionshap-bibliography.bib",
                 "acmart.bib", "acm-jdslogo.png"]
ASSET_DIRS = ("tables", "figures")


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def referenced_files(primary: Path, documents: list[str]) -> tuple[set[Path], set[Path]]:
    """Everything the documents input, follow ``\\input`` chains so nested assets count.

    Returns (resolved, unresolved). ``\\graphicspath{{figures/}}`` is honoured the way TeX
    does it: a bare name that is not in the root is retried under ``figures/``, with and
    without ``.pdf``.
    """
    resolved: set[Path] = set()
    unresolved: set[Path] = set()
    queue = [primary / name for name in documents]
    visited: set[Path] = set()
    while queue:
        tex = queue.pop()
        if not tex.exists() or tex in visited:
            continue
        visited.add(tex)
        resolved.add(tex)
        text = tex.read_text(encoding="utf-8", errors="ignore")
        # TeX looks up graphics in \graphicspath when the name is not found directly.
        graphic_dirs = [Path(d) for group in re.findall(r"\\graphicspath\{((?:\{[^}]*\})+)\}", text)
                        for d in re.findall(r"\{([^}]*)\}", group)] or [Path("figures")]
        found: list[tuple[Path, list[Path]]] = []
        for name in re.findall(r"\\(?:safeinput|input|include)\{([^}#]+)\}", text):
            rel = Path(name if name.endswith(".tex") else name + ".tex")
            found.append((rel, [rel]))
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            rel = Path(name)
            tried = [rel] + [directory / rel for directory in graphic_dirs if not rel.is_absolute()]
            if not rel.suffix:
                tried += [candidate.with_suffix(".pdf") for candidate in list(tried)]
            found.append((rel, tried))
        for bib in re.findall(r"\\bibliography\{([^}]*)\}", text):
            found.extend((Path(key.strip() + ".bib"), [Path(key.strip() + ".bib")])
                         for key in bib.split(","))
        for style in re.findall(r"\\bibliographystyle\{([^}]+)\}", text):
            found.extend((Path(key.strip() + ".bst"), [Path(key.strip() + ".bst")])
                         for key in style.split(","))
        for rel, candidates in found:
            hit = next((primary / candidate for candidate in candidates
                        if not candidate.is_absolute() and (primary / candidate).exists()), None)
            if hit is None:
                unresolved.add(rel)
                continue
            resolved.add(hit)
            if hit.suffix == ".tex":
                queue.append(hit)
    return resolved, unresolved


def build(primary: Path, output: Path) -> int:
    resolved, unresolved = referenced_files(primary, MAIN_DOCUMENTS)
    if unresolved:
        print("the documents reference files that do not exist in the tree:", file=sys.stderr)
        for rel in sorted(unresolved):
            print(f"  {rel}", file=sys.stderr)
        print("Fix the source or add the asset; packing anyway would produce a PDF with "
              "a '[Missing table asset]' box in it.", file=sys.stderr)
        return 1

    names = set()
    payload: dict[str, Path] = {}
    for name in MAIN_DOCUMENTS + CLASS_SUPPORT:
        path = primary / name
        if path.exists():
            names.add(name)
            payload[name] = path
    for sub in ASSET_DIRS:
        for path in sorted((primary / sub).rglob("*")):
            if path.is_file():
                rel = str(path.relative_to(primary))
                names.add(rel)
                payload[rel] = path
    # A referenced file the packer skipped is a bug in this script, not in the sources.
    for path in sorted(resolved):
        rel = str(path.relative_to(primary))
        if rel not in names:
            print(f"internal error: {rel} is referenced but not packed", file=sys.stderr)
            return 1

    stamp = re.search(r"\\newcommand\{\\resultmanifeststamp\}\{([^}]*)\}",
                      (primary / "acmmanuscript.tex").read_text(encoding="utf-8"))
    options = re.findall(r"\\documentclass\s*\[([^\]]*)\]\{acmart\}",
                         (primary / "acmmanuscript.tex").read_text(encoding="utf-8"))
    readme = f"""ActionShap -- ACM TORS review copy (Overleaf project)
======================================================

Generated from the repository at commit {git_revision()} on {date.today():%Y-%m-%d} by
`make overleaf`. Result manifest quoted by both documents: {stamp.group(1) if stamp else 'not found'}.
Re-run `make overleaf` after any change to a .tex, a table or a figure: this project is a
snapshot, and a snapshot that is compiled and downloaded is indistinguishable from a
current one unless the stamp below says otherwise.

Menu -> Settings
  * Main document: acmmanuscript.tex, then switch to supplementary.tex and compile again
    for the second PDF. Both share tables/ and figures/.
  * Compiler: pdfLaTeX, TeX Live 2024 or newer.

Class options as generated: {options[0] if options else '?'} (and the supplement's).
`anonymous` is what a double-blind review copy needs. Drop it only at camera-ready: the
running head and the CRediT paragraph follow it through the \\ifreviewcopy switch in each
preamble, so the author names return with no other edit.

No .bbl is shipped on purpose: the repository copy predates the current text, and a
present-but-stale .bbl lets latexmk skip BibTeX. The bibliography comes from
actionshap-bibliography.bib with ACM-Reference-Format.bst, which are both in this zip.

No compiled PDF is shipped. When you download the two, overwrite
paper-ideas/ActionShap/acmart-primary/acmmanuscript.pdf and .../supplementary.pdf at those
exact paths -- not beside them -- then run `make ready` in the repository. It reads the
compiled text (anonymised title page, the manifest stamp {stamp.group(1) if stamp else ''},
the review-9 panels, no placeholder sentences) and recomputes the deposit's checksums, and
it exits non-zero with the blockers it found.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in payload.items():
            archive.write(path, name)
        archive.writestr("README-OVERLEAF.txt", readme)
        packed = set(archive.namelist())

    # Second pass, from the archive itself: the Overleaf project must resolve on its own.
    with zipfile.ZipFile(output) as archive:
        have = set(archive.namelist())
    dangling = sorted(str(path.relative_to(primary)) for path in resolved
                      if str(path.relative_to(primary)) not in have)
    for name in CLASS_SUPPORT:
        if (primary / name).exists() and name not in have:
            dangling.append(name)
    stale = sorted(n for n in have if n.endswith((".bbl", ".aux", ".log"))
                   or ("/" not in n and n.endswith(".pdf")))
    print(f"{output}: {len(have)} files, {output.stat().st_size / 1e6:.2f} MB")
    print(f"  documents: {[n for n in sorted(have) if n.endswith('.tex') and '/' not in n]}")
    print(f"  assets: {sum(1 for n in have if n.startswith('tables/'))} tables, "
          f"{sum(1 for n in have if n.startswith('figures/'))} figures")
    print(f"  dangling references inside the zip: {dangling or 'none'}")
    print(f"  stale build files: {stale or 'none'}")
    if dangling:
        print("the packed project is not self-contained", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PRIMARY,
                        help="directory holding acmmanuscript.tex (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="zip to write (default: %(default)s)")
    args = parser.parse_args()
    return build(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
