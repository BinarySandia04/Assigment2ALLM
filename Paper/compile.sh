#!/bin/bash
rm main.aux main.bbl main.bcf main.blg main.dvi main.fdb_latexmk main.fls main.log main.old main.pdf main.run.xml main.synctex.gx main.toc
latexmk -f -dvi -pdf -interaction=nonstopmode main.tex
biber main.bcf
latexmk -f -dvi -pdf -interaction=nonstopmode main.tex

