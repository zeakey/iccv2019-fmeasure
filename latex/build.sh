#/bin/bash
set -e
if [ ! -e output ]; then
  mkdir output
fi
if [[ [$(uname) == "Linux"] || [$(uname) == "Darwin"] ]]; then
  LATEX="exlatex"
else
  LATEX="pdflatex"
fi
echo "Building with "$LATEX
"$LATEX" -output-directory output iccv2019fmeasure.tex
bibtex output/iccv2019fmeasure.aux
"$LATEX" -output-directory output iccv2019fmeasure.tex
"$LATEX" -output-directory output iccv2019fmeasure.tex
