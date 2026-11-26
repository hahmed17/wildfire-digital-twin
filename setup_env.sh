#!/usr/bin/env bash
# One-time (and after-dependency-change) setup for wildfire-digital-twin.
#
#   bash setup_env.sh
#
# 1. creates (or updates) the conda environment from environment.yml in ~/envs/wildfire-twin
#    (your home folder survives JupyterHub server restarts; /opt/conda/envs may not)
# 2. installs this repo into it with `pip install -e .` (orchestrator importable everywhere)
# 3. makes src/TestFARSITE and src/lcpmake executable
# 4. registers the Jupyter kernel "Python (wildfire-twin)"
# 5. checks that the main imports work
#
# Options (environment variables):
#   ENV_PATH=~/envs/wildfire-twin   where the environment lives
#   KERNEL_NAME=wildfire-twin       Jupyter kernel name / KERNEL_DISPLAY="Python (wildfire-twin)"
#   CONDA=mamba                     conda / mamba / micromamba (default: mamba if present, else conda)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${ENV_PATH:-$HOME/envs/wildfire-twin}"
KERNEL_NAME="${KERNEL_NAME:-wildfire-twin}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (wildfire-twin)}"
ENV_FILE="$REPO/environment.yml"
CONDA="${CONDA:-$(command -v mamba || command -v conda || true)}"

if [ -z "$CONDA" ]; then
    echo "✗ conda not found. Install Miniforge or run this where conda is available." >&2
    exit 1
fi
[ -f "$ENV_FILE" ] || { echo "✗ $ENV_FILE not found" >&2; exit 1; }

echo "== 1/5 conda environment: $ENV_PATH (using $(basename "$CONDA"))"
mkdir -p "$(dirname "$ENV_PATH")"
if [ -x "$ENV_PATH/bin/python" ]; then
    echo "   exists → updating from environment.yml"
    case "$(basename "$CONDA")" in
        micromamba) "$CONDA" install -y -p "$ENV_PATH" -f "$ENV_FILE" ;;
        *)          "$CONDA" env update -p "$ENV_PATH" -f "$ENV_FILE" ;;
    esac
else
    case "$(basename "$CONDA")" in
        micromamba) "$CONDA" create -y -p "$ENV_PATH" -f "$ENV_FILE" ;;
        *)          "$CONDA" env create -p "$ENV_PATH" -f "$ENV_FILE" ;;
    esac
fi
PY="$ENV_PATH/bin/python"

echo "== 2/5 install this repo (editable)"
if [ -f "$REPO/pyproject.toml" ] || [ -f "$REPO/setup.py" ]; then
    "$PY" -m pip install --quiet -e "$REPO"
else
    echo "   ⚠ no pyproject.toml/setup.py at $REPO: skipped (notebooks find the repo root themselves)"
fi
if [ -f "$REPO/requirements.txt" ]; then
    echo "   also installing requirements.txt"
    "$PY" -m pip install --quiet -r "$REPO/requirements.txt"
fi

echo "== 3/5 FARSITE binaries executable"
for f in "$REPO/src/TestFARSITE" "$REPO/src/lcpmake" "$REPO/install_packages.sh"; do
    [ -f "$f" ] && chmod +x "$f" && echo "   chmod +x ${f#$REPO/}"
done

echo "== 4/5 Jupyter kernel: $KERNEL_DISPLAY"
# The kernel starts without `conda activate`, so pass the environment's paths explicitly
# (PROJ/GDAL data for proj.db, and PATH so shell commands in notebooks use this environment).
"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY" \
    --env PATH "$ENV_PATH/bin:$PATH" \
    --env PROJ_DATA "$ENV_PATH/share/proj" \
    --env PROJ_LIB "$ENV_PATH/share/proj" \
    --env GDAL_DATA "$ENV_PATH/share/gdal" >/dev/null
echo "   registered (select it once per notebook; Jupyter remembers it)"

echo "== 5/5 import check"
cd "$REPO"
PROJ_DATA="$ENV_PATH/share/proj" GDAL_DATA="$ENV_PATH/share/gdal" "$PY" - <<'EOF'
import importlib, sys
ok = True
for mod in ["osgeo.gdal", "geopandas", "pyproj", "shapely", "rasterio", "contextily",
            "pydantic", "langgraph", "nbclient"]:
    try:
        importlib.import_module(mod)
    except Exception as e:
        ok = False; print(f"   ✗ {mod}: {e}")
try:
    import pyproj
    pyproj.CRS.from_epsg(5070)
    from osgeo import osr
    osr.UseExceptions(); s = osr.SpatialReference(); s.ImportFromEPSG(5070)
    print("   ✓ PROJ database found (EPSG:5070 resolves in pyproj and GDAL)")
except Exception as e:
    ok = False; print(f"   ✗ PROJ database: {e}")
try:
    import orchestrator
    from orchestrator import graph, steps, synthetic  # noqa: F401
    print(f"   ✓ orchestrator ({orchestrator.__file__})")
    try:
        from orchestrator import adapter  # noqa: F401
        print("   ✓ orchestrator.adapter (src/ modules import)")
    except ModuleNotFoundError as e:
        ok = False
        print(f"   ✗ orchestrator.adapter: missing module {e.name!r}"
              " → add it to environment.yml (or it's a src/ file that isn't there)")
except Exception as e:
    ok = False; print(f"   ✗ orchestrator: {e}")
print("   all checks passed" if ok else "   some checks failed (see above)")
sys.exit(0 if ok else 1)
EOF

echo
echo "Done. In Jupyter: Kernel → Change Kernel → \"$KERNEL_DISPLAY\" (once per notebook)."
echo "Terminal:  conda activate $ENV_PATH"
