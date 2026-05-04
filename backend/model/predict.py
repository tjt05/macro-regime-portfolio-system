from __future__ import annotations

"""Inference helpers for the saved regime model."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.model.train import MODEL_PATH


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load a persisted PRAIDS model artifact from disk."""
    return joblib.load(model_path)


def predict_regimes(features: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Predict cluster IDs and mapped macro regime labels for each feature row."""
    aligned = features[artifact["feature_columns"]]
    cluster_ids = artifact["model"].predict(aligned)
    regimes = [artifact["regime_mapping"].get(int(cluster), "neutral") for cluster in cluster_ids]
    return pd.DataFrame(
        {
            "cluster": cluster_ids,
            "regime": regimes,
        },
        index=features.index,
    )
