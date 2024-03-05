
paper/document.pdf: paper/document.toc paper/document.tex
	cd paper && pdflatex -halt-on-error document.tex >> /dev/null

paper/document.aux: paper/document.tex
	cd paper && pdflatex -halt-on-error document.tex >> /dev/null

paper/document.blg: paper/document.aux paper/refs.bib
	cd paper && bibtex document.aux >> /dev/null

paper/document.toc: paper/document.blg
	cd paper && pdflatex -halt-on-error document.tex >> /dev/null
