#!/usr/bin/env python3
"""Static manuscript integrity checks that do not require a TeX installation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CITATION_RE = re.compile(r"\\cite(?:p|t|author|year)?\*?\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
SAFE_INPUT_RE = re.compile(r"\\safeinput\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|pageref)\{([^}]+)\}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="../acmart-primary/acmmanuscript.tex")
    parser.add_argument(
        "--bib", default="../acmart-primary/actionshap-bibliography.bib"
    )
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    paper_path = (code_root / args.paper).resolve()
    bib_path = (code_root / args.bib).resolve()
    paper_root = paper_path.parent
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
