# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CommonData:
    """Holds the pre-split, pre-processed train/test arrays.

    The caller is responsible for all data preparation (splitting, scaling,
    encoding). MLKit wrappers only perform shape normalization internally.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class DataBundle:
    """Top-level data container passed to MLKit.

    Attributes:
        common: The four arrays (X_train, y_train, X_test, y_test).
        hyperparameters: Optional per-call overrides that are merged on top of
            config.yaml defaults at model instantiation time. Keys must match
            the model's config.yaml parameter names exactly.
    """

    common: CommonData
    hyperparameters: dict = field(default_factory=dict)
