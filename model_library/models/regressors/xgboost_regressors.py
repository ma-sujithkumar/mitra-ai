# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import numpy as np
from xgboost import XGBRegressor

from core.data_bundle import DataBundle
from models.base import BaseModel


class XGBRegressorWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        self.model = XGBRegressor(**merged_config)
        self.model.fit(data.common.X_train, data.common.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
