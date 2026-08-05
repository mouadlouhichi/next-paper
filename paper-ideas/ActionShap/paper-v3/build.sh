#!/usr/bin/env bash
set -euo pipefail

command -v pdflatex >/dev/null || { echo 'pdflatex is required' >&2; exit 127; }
command -v bibtex >/dev/null || { echo 'bibtex is required' >&2; exit 127; }

pdflatex -interaction=nonstopmode -halt-on-error actionshap.tex
bibtex actionshap
pdflatex -interaction=nonstopmode -halt-on-error actionshap.tex
pdflatex -interaction=nonstopmode -halt-on-error actionshap.tex

if grep -qE 'Citation .* undefined|There were undefined references|undefined citations' actionshap.log; then
  echo 'Build completed with unresolved citations or references' >&2
  exit 2
fi

echo "Built actionshap.pdf with a compiled numbered bibliography."
