# Build the thesis paper draft: latexmk -pdf main.tex
.PHONY: all pdf clean distclean

all: pdf

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -c

distclean:
	latexmk -C
