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

# A venv- or Jupyter-launched shell usually does not carry /Library/TeX/texbin, so
# `make pdf` used to report "no toolchain" on a machine that has MacTeX installed.
# Probe the usual install locations and prepend the first one that exists; an empty
# TEXBIN must not leave an empty PATH entry (that would make `.` a search dir).
TEXBIN := $(firstword $(wildcard /Library/TeX/texbin /opt/homebrew/bin /usr/local/bin \
                        $(addsuffix /bin/x86_64-darwin,$(wildcard /usr/local/texlive/*)) \
                        $(addsuffix /bin/x86_64-linux,$(wildcard /usr/local/texlive/*))))
ifneq ($(strip $(TEXBIN)),)
export PATH := $(TEXBIN):$(PATH)
endif

.PHONY: pdf main supplementary manifest tables check stats clean tools help artifact ready overleaf

help:
	@echo "make tools          report whether a LaTeX toolchain is available"
	@echo "make tables         regenerate every table from the frozen release matrices"
	@echo "make manifest       re-freeze code/results/manifest.json (the hash both PDFs quote)"
	@echo "make stats          tables + manifest"
	@echo "make pdf            rebuild acmmanuscript.pdf and supplementary.pdf"
	@echo "make check          validators, manifest freshness, and the full test suite"
	@echo "make clean          latexmk -C in the paper directory"

tools:
	@command -v pdflatex >/dev/null 2>&1 || { \
	  echo "pdflatex not found in PATH or in the TeX locations this file probes."; \
	  echo "Install it first:"; \
	  echo "  Debian/Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-recommended"; \
	  echo "                 texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra latexmk"; \
	  echo "  macOS:         brew install --cask mactex-no-gui"; \
	  echo "                 MacTeX installs into /Library/TeX/texbin, which this file probes,"; \
	  echo "                 so a new shell is not required -- but latexmk must be present too."; \
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

# Build the deposit archive for OSF/Zenodo: raw runs plus the generated half of the
# artifact (review-9 outputs, tables, manifest, generators, tests, run notebook).
# Reviewer issues 16/17 are about the deposit being the thing the paper was built from,
# so `raw/` alone is not enough; print the archive hash to paste into the data prompt.
artifact: stats
	cd $(CODE) && $(PY) scripts/package_results.py
	@echo "archive sha256 (recorded in the .sha256 sidecar; deliberately NOT quoted in"
	@echo "the documents, because any rebuild changes it and a pasted hash goes stale):"
	@sha256sum $(CODE)/results/release/*.tar.gz | sed 's/^/  /'
	@$(PY) $(CODE)/scripts/make_result_manifest.py --check

stats: tables manifest

# Generate the Overleaf project for the review copy. The zip is a build product, so it
# lands in results/release/build/ (git-ignored): the submission is compiled on Overleaf
# when no TeX distribution is available, and a hand-assembled zip is how a stale .bbl or a
# missing table reaches a compiler that would otherwise have caught it.
overleaf:
	cd $(CODE) && $(PY) scripts/make_overleaf_project.py

# Answer "can this go to the editor right now?" from the files, not from memory: it
# checks the compiled PDFs and the deposit as well as the sources, because a `make check`
# that only reads the .tex passed for a whole round on top of stale PDFs.
ready:
	cd $(CODE) && $(PY) scripts/check_submission.py

check:
	cd $(CODE) && $(PY) scripts/validate_manuscript.py
	cd $(CODE) && $(PY) scripts/validate_cross_table.py
	cd $(CODE) && $(PY) scripts/validate_prose_references.py
	cd $(CODE) && $(PY) scripts/validate_inferential_provenance.py
	cd $(CODE) && $(PY) scripts/make_result_manifest.py --check
	cd $(CODE) && $(PY) -m pytest -q

clean:
	cd $(PAPER) && latexmk -C acmmanuscript.tex supplementary.tex || true
