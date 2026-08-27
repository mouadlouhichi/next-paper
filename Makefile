# Rebuild the ACM TORS submission from source.
#
# The PDFs checked into this repository are *artifacts*: whenever a .tex source,
# a table, or the bibliography changes, they must be rebuilt before submission
# (the review-9 round caught the two PDFs lagging behind the manuscript text).
# `make pdf` does that; `make check` verifies everything, including that the
# PDFs are not older than the sources they were built from.
#
# Paths are resolved from this file's own location, so any of the following work:
#   make -C next-paper stats          from the checkout parent
#   make stats                        from the repository root
#   make -f paper-ideas/ActionShap/../../Makefile pdf

ROOT    := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
PAPER   := $(ROOT)/paper-ideas/ActionShap/acmart-primary
CODE    := $(ROOT)/paper-ideas/ActionShap/code
SCRIPTS := $(CODE)/scripts
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -shell-escape
PY      := python3

.PHONY: pdf main supplementary manifest tables check stats clean tools help

help:
	@echo "make tools          report whether a LaTeX toolchain is available"
	@echo "make tables         regenerate every table from the frozen release matrices"
	@echo "make manifest       re-freeze code/results/manifest.json (the hash both PDFs quote)"
	@echo "make stats          tables + manifest"
	@echo "make pdf            rebuild acmmanuscript.pdf and supplementary.pdf"
	@echo "make check          validators, manifest freshness, and the full test suite"
	@echo "make clean          latexmk -C in the paper directory"

tools:
	@command -v pdflatex >/dev/null || { \
	  echo "pdflatex not found. Install it first:"; \
	  echo "  Debian/Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-recommended \\"; \
	  echo "                 texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra latexmk"; \
	  echo "  macOS:         brew install --cask mactex-no-gui && eval '$$(/usr/libexec/path_helper)'"; \
	  echo "  or use an Overleaf project containing acmart-primary/."; exit 1; }

main: tools
	cd $(PAPER) && $(LATEXMK) acmmanuscript.tex && $(LATEXMK) -pdf acmmanuscript.tex

supplementary: tools
	cd $(PAPER) && $(LATEXMK) supplementary.tex && $(LATEXMK) -pdf supplementary.tex

pdf: main supplementary
	@echo "rebuilt acmmanuscript.pdf and supplementary.pdf"

# Regenerate every table from the frozen release matrices, then re-freeze the
# content hash that both documents quote.
# review3_statistics.tex also contains blocks appended by hand, so its
# generator only *verifies* the rows it owns (--check) instead of rewriting the
# file; the review-9 generator owns its files outright.
tables:
	cd $(CODE) && $(PY) scripts/make_review3_stats.py --check
	cd $(CODE) && $(PY) scripts/make_review9_stats.py
	cd $(CODE) && $(PY) scripts/make_review5_tables.py

manifest:
	cd $(CODE) && $(PY) scripts/make_result_manifest.py

stats: tables manifest

check:
	cd $(CODE) && $(PY) scripts/validate_manuscript.py
	cd $(CODE) && $(PY) scripts/validate_cross_table.py
	cd $(CODE) && $(PY) scripts/make_result_manifest.py --check
	cd $(CODE) && $(PY) -m pytest -q

clean:
	cd $(PAPER) && latexmk -C acmmanuscript.tex supplementary.tex || true
