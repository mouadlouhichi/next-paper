#!/usr/bin/env python3
"""Check that cross-document pointers in the manuscript can actually be resolved.

Supplementary *table* numbers are assigned by the float counter at compile time, so a hand-typed
``Table~S25`` in the main text is unverifiable in the source and had already drifted out of the
declared range twice.  The policy this validator pins is therefore:

  * supplementary pointers in the main document name a *section* (or a table that section titles
    itself), and that section exists;
  * the supplement's declared ``Sections S1--Sn`` subtitle matches the real section count;
  * neither document says ``Appendix S<n>`` for a section-numbered supplement.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "acmart-primary" / "acmmanuscript.tex"
SUPP = ROOT / "acmart-primary" / "supplementary.tex"

SECTION = re.compile(r"\\section\{([^{}]*)\}")
DECLARED = re.compile(r"\\subtitle\{Sections S1--S(\d+)\}")
SUPP_POINTER = re.compile(r"Supplementary (Section|Sections)\s*~?\s*((?:S\d+)(?:\s*(?:and|,)\s*S\d+)*)")
APPENDIX_S = re.compile(r"Appendix~?\s*S\d+")
TABLE_POINTER = re.compile(r"Supplementary Tables?\s*~?\s*(S\d+)(?:--(S\d+))?")
TITLE_TABLES = re.compile(r"Tables?\s*S(\d+)(?:--S(\d+))?")
NUMBER = re.compile(r"S(\d+)")


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def main() -> int:
    main_tex = MAIN.read_text(encoding="utf-8")
    supp_tex = SUPP.read_text(encoding="utf-8")
    problems: list[str] = []

    sections = SECTION.findall(supp_tex)
    n_sections = len(sections)

    declared = DECLARED.findall(supp_tex)
    if not declared:
        fail(problems, "supplementary.tex declares no 'Sections S1--Sn' subtitle to check")
    else:
        for value in declared:
            if int(value) != n_sections:
                fail(problems,
                     f"supplement subtitle declares Sections S1--S{value} but the file has "
                     f"{n_sections} \\section commands")

    titles = {i + 1: title.strip() for i, title in enumerate(sections)}
    declared_tables: set[int] = set()
    for title in sections:
        for low, high in TITLE_TABLES.findall(title):
            declared_tables.update(range(int(low), int(high or low) + 1))
    used: set[int] = set()
    for kind, numbers in SUPP_POINTER.findall(main_tex):
        for number in NUMBER.findall(numbers):
            index = int(number)
            used.add(index)
            if not 1 <= index <= n_sections:
                fail(problems,
                     f"main text points at Supplementary {kind} S{index}, outside S1--S{n_sections}")
    for low, high in TABLE_POINTER.findall(main_tex):
        for index in range(int(low[1:]), int((high or low)[1:]) + 1):
            if index not in declared_tables:
                fail(problems,
                     f"main text points at Supplementary Table S{index}, which no supplementary "
                     f"section title declares (declared: {sorted(declared_tables)}); supplement "
                     f"table floats are numbered at compile time, so point at the section instead "
                     f"or at a table its own section title names")
    for path, text in ((MAIN, main_tex), (SUPP, supp_tex)):
        for match in APPENDIX_S.findall(text):
            fail(problems, f"{path.name} says '{match}' although the supplement numbers sections S1--Sn")

    unpointed = sorted(set(titles) - used)
    print(f"supplementary sections: {n_sections}; referenced from the main text: "
          f"{sorted(used) if used else 'none'}")
    for index in sorted(used):
        print(f"  S{index}: {titles[index]}")
    if unpointed:
        print(f"  (not cited from the main text: {', '.join('S%d' % i for i in unpointed)})")

    if problems:
        print("\n".join(f"PROBLEM: {p}" for p in problems), file=sys.stderr)
        return 1
    print("prose references: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
