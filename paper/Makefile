# Build the thesis paper draft: latexmk -pdf main.tex
.PHONY: all pdf dist clean distclean

all: dist

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

# Outlook/mail-safe copy: re-embed every font fully (no subsetting) via ghostscript.
dist: pdf
	gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.6 \
	  -dEmbedAllFonts=true -dSubsetFonts=false -dPDFSETTINGS=/prepress \
	  -sOutputFile=Thesis_Paper.pdf main.pdf

clean:
	latexmk -c

distclean:
	latexmk -C
