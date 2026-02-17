#!/bin/bash

{
    set -euo pipefail

    # initialize conda for non-interactive shells
    # source "$(conda info --base)/etc/profile.d/conda.sh"

    # conda activate base
    # conda install -c conda-forge gdal -y 
    
    # pip -q install -r requirements.txt 
    #!/usr/bin/env bash
    
    echo "PWD: $(pwd)"
    echo "Python: $(python -c 'import sys; print(sys.executable)')"
    
    # Ensure conda commands work in non-interactive shells (safe even if already works)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # conda update -n base -c conda-forge conda
    conda activate base
    
    echo "After conda activate:"
    which python
    /opt/conda/bin/python -V
    /opt/conda/bin/python -m pip -V
    
    conda install -c conda-forge -y gdal
    
    /opt/conda/bin/python -m pip install --upgrade pip
    /opt/conda/bin/python -m pip install -r requirements.txt
    
} >install.log 2>&1
