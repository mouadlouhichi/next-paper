MAIN=main

all: pdf

pdf:
	pdflatex $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

zip:
	cd .. && zip -r CoalGameRec_springer_package.zip paper_package -x 'paper_package/*.pdf' 'paper_package/*.aux' 'paper_package/*.log' 'paper_package/*.bbl' 'paper_package/*.blg' 'paper_package/*.out'
