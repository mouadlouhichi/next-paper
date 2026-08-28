#!/usr/bin/env python3
"""Static manuscript integrity checks that do not require a TeX installation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CITATION_RE = re.compile(r"\\cite(?:p|t|author|year)?\*?\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
SAFE_INPUT_RE = re.compile(r"\\safeinput\{([^}#]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|pageref)\{([^}]+)\}")
INPUT_RE = re.compile(r"\\(?:safeinput|input)\{([^}#]+)\}")
NUMERIC_RE = re.compile(r"[-+]?[0-9]+\.[0-9]+")


def referenced_labels(document: Path) -> set[str]:
    """Labels defined by a document plus every table file it inputs."""
    text = document.read_text()
    labels = set(LABEL_RE.findall(text))
    root = document.parent
    for relative in set(INPUT_RE.findall(text)):
        target = root / relative
        if target.is_file():
            labels |= set(LABEL_RE.findall(target.read_text()))
    return labels


def pdf_creation_date(pdf: Path):
    """The /CreationDate embedded in a PDF, if it can be read."""
    try:
        raw = pdf.read_bytes()
        # The information dictionary usually lives in the trailer, so the tail is
        # searched before the head.
        head, tail = raw[:262144], raw[-262144:]
        pattern = rb"/CreationDate\s*\(\s*(D:)?(\d{14})"
        match = re.search(pattern, tail) or re.search(pattern, head)
        if not match:
            return None
        digits = match.group(2).decode()
        return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def newest_source_time(root: Path, sources: list[Path]):
    """Latest change time among the sources and the tables they input.

    Git commit dates are preferred over file mtimes: a fresh clone resets every
    mtime to checkout time, which would hide a stale PDF.
    """
    repo = root
    while not (repo / ".git").exists() and repo != repo.parent:
        repo = repo.parent
    moments: list[datetime] = []
    for source in sources:
        if not source.exists():
            continue
        candidates = [source] + [
            root / relative
            for relative in sorted(set(INPUT_RE.findall(source.read_text())))
            if (root / relative).is_file()
        ]
        for candidate in candidates:
            stamp = None
            if (repo / ".git").exists():
                try:
                    out = subprocess.run(
                        ["git", "-C", str(repo), "log", "-1", "--format=%ct", "--", str(candidate)],
                        capture_output=True,
                        text=True,
                        check=False,
                    ).stdout.strip()
                    if out.isdigit():
                        stamp = datetime.fromtimestamp(int(out), tz=timezone.utc)
                except Exception:
                    stamp = None
            if stamp is None:
                stamp = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
            moments.append(stamp)
    return max(moments) if moments else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="../acmart-primary/acmmanuscript.tex")
    parser.add_argument(
        "--bib", default="../acmart-primary/actionshap-bibliography.bib"
    )
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--supplement", default="../acmart-primary/supplementary.tex")
    parser.add_argument("--mirror", default="../actionshap-ipm")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    paper_path = (code_root / args.paper).resolve()
    bib_path = (code_root / args.bib).resolve()
    paper_root = paper_path.parent
    mirror_root = (code_root / args.mirror).resolve()
    tex = paper_path.read_text()
    bib = bib_path.read_text()

    errors: list[str] = []
    warnings: list[str] = []
    citations = {
        key.strip()
        for group in CITATION_RE.findall(tex)
        for key in group.split(",")
        if key.strip()
    }
    bibliography_keys = set(BIB_KEY_RE.findall(bib))
    if missing := sorted(citations - bibliography_keys):
        errors.append(f"missing bibliography keys: {missing}")
    if unused := sorted(bibliography_keys - citations):
        warnings.append(f"unused bibliography keys: {unused}")
    if not bibliography_keys:
        errors.append("bibliography contains no entries")
    bib_stem = bib_path.stem
    if f"\\bibliography{{{bib_stem}}}" not in tex:
        errors.append(
            f"manuscript does not point to the pinned bibliography ({bib_stem})"
        )

    begins = Counter(BEGIN_RE.findall(tex))
    ends = Counter(END_RE.findall(tex))
    if begins != ends:
        errors.append(
            f"unbalanced environments: begin={dict(begins)}, end={dict(ends)}"
        )
    if tex.count("{") != tex.count("}"):
        errors.append("raw brace counts differ")

    generated_labels: list[str] = []
    for relative in SAFE_INPUT_RE.findall(tex):
        asset = paper_root / relative
        if asset.exists():
            generated_labels.extend(LABEL_RE.findall(asset.read_text()))
    labels = LABEL_RE.findall(tex) + generated_labels
    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        errors.append(f"duplicate manuscript labels: {duplicate_labels}")
    missing_refs = sorted(set(REF_RE.findall(tex)) - set(labels))
    if missing_refs:
        errors.append(f"unresolved manuscript references: {missing_refs}")
    if "ActionShap:" not in tex or "\\title{" not in tex:
        errors.append("canonical ActionShap title is missing")
    if "\\keywords{" not in tex:
        errors.append("Keywords metadata is missing")
    if "paper-v2" in tex:
        errors.append("stale paper-v2 path remains in canonical manuscript")
    if any(token in tex for token in ("0.749152", "0.918817", "0.000584487")):
        errors.append("invalidated schema-v1 pilot values remain in canonical manuscript")

    missing_assets = [
        relative
        for relative in SAFE_INPUT_RE.findall(tex)
        if not (paper_root / relative).exists()
    ]
    if missing_assets:
        errors.append(f"missing generated assets: {missing_assets}")

    validation_path = paper_root / "final" / "manifests" / "validation_report.json"
    validation = (
        json.loads(validation_path.read_text())
        if validation_path.exists()
        else {"status": "MISSING"}
    )
    if args.require_final and validation.get("status") != "PASS":
        errors.append(
            f"final asset validation is {validation.get('status')}, not PASS"
        )

    # ---- review-9 additions: cross-document and artifact integrity ----
    supplement_path = (code_root / args.supplement).resolve()
    if supplement_path.exists():
        supplement = supplement_path.read_text()
        main_labels = referenced_labels(paper_path)
        supp_labels = referenced_labels(supplement_path)
        dangling = sorted(
            reference
            for reference in set(REF_RE.findall(tex))
            if reference not in main_labels and not reference.startswith("app:")
        )
        if dangling:
            errors.append(f"main-paper references without a label in the main paper: {dangling}")
        cross = sorted(
            reference for reference in set(REF_RE.findall(supplement))
            if reference not in supp_labels | main_labels and not reference.startswith("app:")
        )
        if cross:
            errors.append(f"supplement references that resolve in neither document: {cross}")
        for relative in sorted(set(INPUT_RE.findall(supplement)) | set(SAFE_INPUT_RE.findall(supplement))):
            if not (supplement_path.parent / relative).exists():
                errors.append(f"supplement input does not exist: {relative}")
        # A generated table that no document inputs is invisible in the PDFs,
        # which is how the review-9 round found results reported only in source.
        # \\input paths may or may not carry the extension, so normalize both sides.
        def _normalized(assets: set[str]) -> set[str]:
            return {a if a.endswith(".tex") else a + ".tex" for a in assets}

        # Any of the four documents can input a generated table, so the orphan
        # check has to look at all of them: reporting a table as uninput while the
        # IPM main file \input's it is exactly the cross-document blindness that
        # review-9 issue 17 complained about.
        sources = [tex, supplement]
        for extra in (
            paper_root.parent / "actionshap-ipm" / "actionshap.tex",
            paper_root.parent / "actionshap-ipm" / "supplementary.tex",
        ):
            if extra.exists():
                sources.append(extra.read_text())
        included = _normalized(
            set().union(*[
                set(INPUT_RE.findall(s)) | set(SAFE_INPUT_RE.findall(s)) for s in sources
            ])
        )
        orphans = sorted(
            f"tables/{path.name}"
            for path in (paper_root / "tables").glob("*.tex")
            if f"tables/{path.name}" not in included
        )
        if orphans:
            warnings.append(
                f"generated tables that neither document inputs: {orphans}"
            )
        shared = sorted(
            {p.name for p in (paper_root / "tables").glob("*.tex")}
            & {p.name for p in (mirror_root / "tables").glob("*.tex")}
        ) if mirror_root.is_dir() else []
        drift = [
            name for name in shared
            if NUMERIC_RE.findall((paper_root / "tables" / name).read_text())
            != NUMERIC_RE.findall((mirror_root / "tables" / name).read_text())
        ]
        if drift:
            errors.append(
                f"main/mirror table copies carry different numbers (Issue 17 drift): {drift}"
            )

    manifest_path = code_root / "results" / "manifest.json"
    if manifest_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "actionshap_manifest", code_root / "scripts" / "make_result_manifest.py"
            )
            manifest_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(manifest_module)
            current = manifest_module.build()
            for name in ("acmmanuscript.tex", "supplementary.tex"):
                found = manifest_module.tex_stamp(paper_root / name)
                if found != current["manifest_stamp"]:
                    errors.append(
                        f"{name}: result-manifest stamp {found!r} != current "
                        f"{current['manifest_stamp']!r}; run make_result_manifest.py"
                    )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"result manifest could not be verified: {exc}")
    else:
        warnings.append("code/results/manifest.json missing; run make_result_manifest.py")

    for name in ("acmmanuscript.pdf", "supplementary.pdf"):
        pdf = paper_root / name
        if not pdf.exists():
            continue
        stamp = pdf_creation_date(pdf)
        source = paper_path if name.startswith("acm") else supplement_path
        newest = newest_source_time(paper_root, [source])
        if stamp is not None and newest is not None and stamp < newest:
            warnings.append(
                f"{name} was built at {stamp} but its sources changed at {newest}: rebuild before submission"
            )

    report = {
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "paper": str(paper_path),
        "citations": len(citations),
        "bibliography_entries": len(bibliography_keys),
        "asset_validation": validation.get("status"),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
