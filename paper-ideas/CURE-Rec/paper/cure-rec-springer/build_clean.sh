#!/usr/bin/env bash
# Clean deterministic Springer build. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"
rm -f cure-rec.aux cure-rec.bbl cure-rec.blg cure-rec.log cure-rec.out cure-rec.toc cure-rec.pdf
pdflatex -interaction=nonstopmode -halt-on-error cure-rec.tex
bibtex cure-rec
pdflatex -interaction=nonstopmode -halt-on-error cure-rec.tex
pdflatex -interaction=nonstopmode -halt-on-error cure-rec.tex
# Fail the build if common unresolved-publication artifacts remain.
if grep -Eq 'Table \?\?|\?\?\?' cure-rec.log; then
  echo 'Unresolved table/reference artifact found in cure-rec.log' >&2
  exit 1
fi
echo "Built $(pwd)/cure-rec.pdf"
