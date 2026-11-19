"""Orchestrator package.

Digital twin source code lives in wildfire-digital-twin/src as top-level modules (enkf, farsite, config, firemap, geometry) that import each other by plain name. Put src/ on the import path whenever this package is imported, so `import enkf` works from anywhere: the repo root, notebooks, pytest, or scripts.
"""
import sys as _sys
from pathlib import Path as _Path

_SRC = _Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))