paper-jasa.pdf: learning.bib paper-jasa.tex
	rm -f *.pdf
	pdflatex -shell-escape paper-jasa
	pdflatex -shell-escape paper-jasa
	rm -f *.log *.out *.aux *.bbl *.blg

full: learning.bib paper-jasa.tex
	pdflatex -shell-escape paper-jasa
	bibtex paper-jasa
	pdflatex -shell-escape paper-jasa
	pdflatex -shell-escape paper-jasa
	rm -f *.log *.out *.aux *.bbl *.blg

clean:
	rm -f *.log *.out *.aux *.bbl *.blg *.data paper-jasa.pdf
