"""SHAP visualization package: summary, beeswarm, and feature importance plot generators.

matplotlib Agg backend is configured at package-import time so it is always set
before any matplotlib.pyplot import in the submodules, regardless of which generator
class the caller imports first.
"""

import matplotlib

matplotlib.use("Agg")
