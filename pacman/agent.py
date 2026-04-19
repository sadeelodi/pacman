"""Model-backed agent that predicts Pac-Man moves."""

import os

import joblib
import pandas as pd

from .features import extract_features


class RandomForestAgent:
    """Thin wrapper around a saved sklearn model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, scene) -> str | None:
        if not self.available:
            return None

        features = extract_features(scene)
        if hasattr(self.model, "feature_names_in_"):
            ordered = {name: features[name] for name in self.model.feature_names_in_}
        else:
            ordered = features

        frame = pd.DataFrame([ordered])
        return str(self.model.predict(frame)[0])
