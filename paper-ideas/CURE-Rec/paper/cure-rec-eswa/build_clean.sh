#!/usr/bin/env bash
# Clean deterministic ESWA (elsarticle) build. Run from this directory.
# Requires a full TeX Live / MacTeX installation.
set -euo pipefail
cd "$(dirname "$0")"
JOB=cure-rec-eswa
rm -f $JOB.aux $JOB.bbl $JOB.blg $JOB.log $JOB.out $JOB.toc $JOB.pdf
pdflatex -interaction=nonstopmode -halt-on-error $JOB.tex
bibtex $JOB
pdflatex -interaction=nonstopmode -halt-on-error $JOB.tex
pdflatex -interaction=nonstopmode -halt-on-error $JOB.tex
# Fail the build if unresolved publication artifacts or width violations remain.
if grep -Eq 'Table \?\?|\?\?\?|There were undefined references|Reference .* undefined|Citation .* undefined|multiply defined|Overfull \\hbox|Overfull \\vbox' $JOB.log $JOB.blg; then
  echo 'Unresolved reference, duplicate label, or overfull box found in build logs' >&2
  exit 1
fi
echo "Built $(pwd)/$JOB.pdf"
