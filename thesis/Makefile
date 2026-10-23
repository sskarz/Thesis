# Build the CSULA-format thesis: latexmk -pdf main.tex
.PHONY: all pdf dist clean distclean

all: dist

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

# Outlook/mail-safe copy: re-embed every font fully (no subsetting) via ghostscript.
dist: pdf
	gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.6 \
	  -dEmbedAllFonts=true -dSubsetFonts=false -dPDFSETTINGS=/prepress \
	  -sOutputFile=Thesis_CSULA_Format.pdf main.pdf

clean:
	latexmk -c

distclean:
	latexmk -C
