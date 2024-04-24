
paper/document.pdf: paper/document.toc paper/document.tex
	cd paper && pdflatex --shell-escape -halt-on-error document.tex >> /dev/null

paper/document.aux: paper/document.tex paper/svg-graphs
	cd paper && pdflatex --shell-escape -halt-on-error document.tex >> /dev/null

paper/document.blg: paper/document.aux paper/refs.bib
	cd paper && bibtex document.aux >> /dev/null

paper/document.toc: paper/document.blg
	cd paper && pdflatex --shell-escape -halt-on-error document.tex >> /dev/null


paper/svg-graphs: data.csv analysis.R
	rm -r paper/svg-graphs
	mkdir paper/svg-graphs
	cat analysis.R | R --no-save
