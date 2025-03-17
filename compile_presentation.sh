#!/bin/bash

# Compile script for face masks presentation
# This script will compile the presentation with XeLaTeX three times
# to ensure proper reference resolution and TOC generation

echo "Compiling Face Masks Market Analysis Presentation..."
echo "First XeLaTeX run..."
xelatex -interaction=nonstopmode face_masks_presentation.tex

echo "Second XeLaTeX run..."
xelatex -interaction=nonstopmode face_masks_presentation.tex

echo "Third XeLaTeX run..."
xelatex -interaction=nonstopmode face_masks_presentation.tex

echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.synctex.gz

echo "Done! Presentation PDF is ready: face_masks_presentation.pdf"