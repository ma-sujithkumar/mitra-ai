"""dataset2Vec warm-start package.

This package's internal modules import their siblings as top-level packages
(``from d2v_core.X import ...``) and the shared model_library (``from ml_kit
import ...``, ``from core.X import ...``, ``from metrics.X import ...``). To keep
those imports resolving when the package is imported under its new home
(``backend.agents.dataset2vec``, e.g. from the orchestrator), bootstrap both this
directory and the repo-root ``model_library`` onto ``sys.path`` at import time.
No paths are hardcoded; everything resolves relative to this file's location.
"""
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
# dataset2vec -> agents -> backend -> repo root
_MODEL_LIBRARY_ROOT = _PACKAGE_DIR.parents[2] / "model_library"

for _bootstrap_path in (_PACKAGE_DIR, _MODEL_LIBRARY_ROOT):
    _bootstrap_path_str = str(_bootstrap_path)
    if _bootstrap_path_str not in sys.path:
        sys.path.insert(0, _bootstrap_path_str)
