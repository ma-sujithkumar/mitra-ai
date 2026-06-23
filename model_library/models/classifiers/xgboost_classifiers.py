# @Authored by Claude Sonnet 4.6, Co-Authored by Sujith M A, Created 2026-05-31, Last Updated 2026-05-31
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from core.data_bundle import DataBundle
from models.base import BaseModel


class XGBClassifierWrapper(BaseModel):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.label_encoder: LabelEncoder = LabelEncoder()

    def train(self, data: DataBundle) -> None:
        merged_config = self._merge_hyperparameter_overrides(data)
        # use_label_encoder is deprecated in recent xgboost — remove silently if present
        merged_config.pop("use_label_encoder", None)
        # XGBoost requires class labels to be consecutive integers 0..N-1.
        # LabelEncoder remaps arbitrary integer labels (e.g. {0,1,3,...,14}) to that range.
        y_train_encoded = self.label_encoder.fit_transform(data.common.y_train)
        self.model = XGBClassifier(**merged_config)
        self.model.fit(data.common.X_train, y_train_encoded)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Decode predictions back to original labels so downstream metrics are correct.
        encoded_predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded_predictions)
