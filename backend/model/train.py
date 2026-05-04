from __future__ import annotations

"""Model training and persistence.

PRAIDS uses KMeans as an interpretable unsupervised regime detector. Training is
cached in `artifacts/regime_model.joblib` so normal app runs use `predict`
against the saved model. Use `force=True` when the feature set changes or when
you intentionally want to refresh cluster assignments with newer data.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.assets import TRADABLE_ASSETS
from backend.macro.regime_interpreter import derive_regime_mapping


MODEL_PATH = Path("artifacts/regime_model.joblib")
ARTIFACT_VERSION = "praids_v3_macro_multi_asset"


def train_regime_model(
    features: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
    model_path: str | Path = MODEL_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """Train or load the KMeans regime model and derived macro mapping."""
    path = Path(model_path)
    if path.exists() and not force:
        artifact = joblib.load(path)
        if _artifact_is_compatible(artifact, features):
            return artifact

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)),
        ]
    )
    model.fit(features)

    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features.columns,
    )
    regime_mapping = derive_regime_mapping(centroids)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "asset_universe": TRADABLE_ASSETS,
        "model": model,
        "feature_columns": list(features.columns),
        "centroids": centroids,
        "regime_mapping": regime_mapping,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return artifact


def _artifact_is_compatible(artifact: dict[str, Any], features: pd.DataFrame) -> bool:
    """Check whether a saved model matches the current code/data contract."""
    return (
        artifact.get("artifact_version") == ARTIFACT_VERSION
        and artifact.get("asset_universe") == TRADABLE_ASSETS
        and set(artifact.get("feature_columns", [])) == set(features.columns)
    )
