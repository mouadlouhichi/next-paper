#!/usr/bin/env python3
"""Static manuscript integrity checks that do not require a TeX installation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CITATION_RE = re.compile(r"\\cite[pt]?\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
END_RE = re.compile(r"\\end\{([^}]+)\}")
SAFE_INPUT_RE = re.compile(r"\\safeinput\{([^}]+)\}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="../paper/paper.tex")
    parser.add_argument("--bib", default="../paper/paper.bib")
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
        key.strip() for group in CITATION_RE.findall(tex) for key in group.split(",")
    }
    bibliography_keys = set(BIB_KEY_RE.findall(bib))
    if missing := sorted(citations - bibliography_keys):
        errors.append(f"missing bibliography keys: {missing}")
    if unused := sorted(bibliography_keys - citations):
        warnings.append(f"unused bibliography keys: {unused}")

    begins = Counter(BEGIN_RE.findall(tex))
    ends = Counter(END_RE.findall(tex))
    if begins != ends:
        errors.append(
            f"unbalanced environments: begin={dict(begins)}, end={dict(ends)}"
        )
    if tex.count("{") != tex.count("}"):
        errors.append("raw brace counts differ")
    if "ActionShap: Intervention-Grounded" not in tex:
        errors.append("canonical title is missing")
    if "legacy_pilot" in tex and "legacy\\_pilot" not in tex:
        errors.append("unescaped legacy_pilot path in manuscript text")
    if any(token in tex for token in ("0.749152", "0.918817", "0.000584487")):
        errors.append(
            "invalidated schema-v1 pilot values remain in canonical manuscript"
        )

    missing_assets = [
        relative
        for relative in SAFE_INPUT_RE.findall(tex)
        if not (paper_root / relative).exists()
    ]
    if missing_assets:
        warnings.append(f"generated tables pending: {missing_assets}")

    validation_path = paper_root / "final" / "manifests" / "validation_report.json"
    validation = (
        json.loads(validation_path.read_text())
        if validation_path.exists()
        else {"status": "MISSING"}
    )
    pending_count = tex.count("\\pending") - 1  # subtract macro definition
    if pending_count:
        warnings.append(f"{pending_count} manuscript placeholders remain")
    if args.require_final:
        if validation.get("status") != "PASS":
            errors.append(
                f"final asset validation is {validation.get('status')}, not PASS"
            )
        if pending_count:
            errors.append("final manuscript still contains placeholders")
        if missing_assets:
            errors.append("final manuscript tables are missing")

    report = {
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "paper": str(paper_path.relative_to(code_root.parent.parent.parent)),
        "citations": len(citations),
        "bibliography_entries": len(bibliography_keys),
        "pending_placeholders": pending_count,
        "asset_validation": validation.get("status"),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
