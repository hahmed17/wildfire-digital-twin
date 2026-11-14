#!/bin/bash
set -euo pipefail
mkdir -p src

{
    source "$(conda info --base)/etc/profile.d/conda.sh"
    ENV=gdal_env

    # Use mamba if available (much faster solver), otherwise conda's libmamba solver
    if command -v mamba >/dev/null 2>&1; then
        INSTALLER="mamba"
    else
        INSTALLER="conda --solver=libmamba"
    fi

    # Create a fresh env with the hard-to-build geo packages from conda-forge
    if ! conda env list | grep -q "^${ENV} "; then
        $INSTALLER create -n "$ENV" -c conda-forge --override-channels -y \
            python=3.11 gdal geopandas shapely contextily basemap ipykernel
    fi

    conda activate "$ENV"
    echo "Using: $(which python)"

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

    # Register the kernel the notebook expects
    python -m ipykernel install --user --name gdal_env --display-name "Python (gdal_env)"

    echo "Done."
} 2>&1 | tee src/install.log