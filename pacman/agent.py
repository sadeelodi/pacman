"""Model-backed agent that predicts Pac-Man moves."""

import os
from pathlib import Path

import joblib
import pandas as pd

from .constants import DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_UP
from .board_analysis import choose_board_move
from .features import extract_features


class RandomForestAgent:
    """Thin wrapper around a saved sklearn model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.scaler_path = self._default_scaler_path(model_path)
        self.model = None
        self.scaler = None

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            if hasattr(self.model, "set_params") and hasattr(self.model, "n_jobs"):
                self.model.set_params(n_jobs=1)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    @staticmethod
    def _default_scaler_path(model_path: str) -> str:
        path = Path(model_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / path

        scaler_dir = path.parent.parent / "scaler"
        return str(scaler_dir / f"{path.stem}_scaler.joblib")

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
        if self.scaler is not None:
            if hasattr(self.scaler, "feature_names_in_"):
                frame = frame[list(self.scaler.feature_names_in_)]
            scaled = self.scaler.transform(frame)
            frame = pd.DataFrame(scaled, columns=frame.columns, index=frame.index)
        if hasattr(self.model, "feature_names_in_"):
            frame = frame[list(self.model.feature_names_in_)]
        return str(self.model.predict(frame)[0])


class HeuristicAgent:
    """Rule-based agent that follows board heuristics instead of a trained model."""

    def predict(self, scene) -> str | None:
        move, _ = choose_board_move(scene)
        if move in (DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT):
            return move
        return None
