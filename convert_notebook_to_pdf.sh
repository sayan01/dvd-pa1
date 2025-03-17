#!/bin/bash

echo "Converting Face Masks Analysis Jupyter notebook to PDF..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    echo "Please install pip first and try again"
    exit 1
fi

# Install required packages if not already installed
echo "Checking and installing required packages..."
pip install nbconvert jupyter jupyterlab --quiet || {
    echo "Error installing packages. Please try manually:"
    echo "pip install nbconvert jupyter jupyterlab"
    exit 1
}

# Check for LaTeX installation (required for PDF conversion)
if ! command -v pdflatex &> /dev/null; then
    echo "Warning: pdflatex is not installed. Will attempt HTML conversion instead."
    
    # Convert to HTML
    echo "Converting notebook to HTML..."
    jupyter nbconvert --to html face_masks_analysis.ipynb || {
        echo "Error converting to HTML. Please check if the notebook exists and is valid."
        exit 1
    }
    
    echo "Successfully converted to HTML: face_masks_analysis.html"
    echo "Note: For PDF conversion, please install a LaTeX distribution like TeX Live or MiKTeX"
else
    # Convert to PDF
    echo "Converting notebook to PDF..."
    jupyter nbconvert --to pdf face_masks_analysis.ipynb || {
        echo "Error converting to PDF. Trying alternative approach..."
        
        # Alternative: Convert to HTML first, then use wkhtmltopdf if available
        if command -v wkhtmltopdf &> /dev/null; then
            echo "Attempting conversion via HTML and wkhtmltopdf..."
            jupyter nbconvert --to html face_masks_analysis.ipynb && \
            wkhtmltopdf face_masks_analysis.html face_masks_analysis.pdf && \
            echo "Successfully converted to PDF: face_masks_analysis.pdf" || \
            echo "Error in alternative conversion approach"
        else
            echo "Please install a full LaTeX distribution or wkhtmltopdf for PDF conversion"
            exit 1
        fi
    }
fi

# Also create a Python code PDF
echo "Generating syntax-highlighted PDF of Python code..."
if command -v enscript &> /dev/null && command -v ps2pdf &> /dev/null; then
    enscript -E --color -q -p - face_masks_analysis.py | ps2pdf - face_masks_analysis_code.pdf && \
    echo "Successfully created Python code PDF: face_masks_analysis_code.pdf" || \
    echo "Error creating Python code PDF"
else
    # Alternative method using highlight if available
    if command -v highlight &> /dev/null; then
        highlight -O latex face_masks_analysis.py -o face_masks_analysis_code.tex && \
        pdflatex face_masks_analysis_code.tex && \
        echo "Successfully created Python code PDF: face_masks_analysis_code.pdf" && \
        rm -f face_masks_analysis_code.aux face_masks_analysis_code.log || \
        echo "Error creating Python code PDF using highlight"
    else
        echo "For code syntax highlighting in PDF, please install enscript and ghostscript or highlight"
    fi
fi

echo "Conversion process completed"