"""Ensure repository-local packages are importable when pytest is invoked directly."""

from __future__ import annotations

import sys
from pathlib import Path

# tests -> model_selection -> agents -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
