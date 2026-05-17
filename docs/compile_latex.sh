#!/usr/bin/env bash
set -euo pipefail

TEXBIN="/home/tzh03/texlive/2026/bin/x86_64-linux"
export PATH="$TEXBIN:$PATH"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/report docs/report/report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/report docs/report/report.tex

pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/slides docs/slides/slides.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/slides docs/slides/slides.tex
