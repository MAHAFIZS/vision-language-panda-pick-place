# -*- coding: utf-8 -*-
"""
train_nn_bc_baseline.py

Phase 8B:
Nearest-Neighbor Behavior Cloning baseline for VLA action prediction.

Why this baseline?
The RandomForest regression baseline can average categorical values such as
object_id and target_id. With a tiny dataset, this can decode to wrong objects.

Nearest Neighbor avoids that by retrieving the most similar demonstration
and reusing its stored action vector.

Run:
    python3 train_nn_bc_baseline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataset.dataset_loader import build_behavior_cloning_arrays


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "nn_bc_policy.pkl"


def train_nearest_neighbor_policy(X: np.ndarray) -> NearestNeighbors:
    """
    Train nearest-neighbor retrieval model on demo input features.
    """
    model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )
    model.fit(X)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 8B: Nearest-Neighbor BC Baseline ---")

    X, Y, demos = build_behavior_cloning_arrays("datasets")

    print(f"Loaded demos: {len(demos)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    model = train_nearest_neighbor_policy(X)

    distances, indices = model.kneighbors(X)

    print("\n--- Training-set retrieval sanity check ---")
    for i, (dist, idx) in enumerate(zip(distances[:, 0], indices[:, 0]), start=1):
        command = demos[i - 1]["data"]["language"]
        retrieved_command = demos[idx]["data"]["language"]

        print(f"\nQuery demo {i}")
        print(f"Input command:     {command}")
        print(f"Retrieved demo:    demo_{idx + 1:04d}")
        print(f"Retrieved command: {retrieved_command}")
        print(f"Distance:          {dist:.6f}")

    bundle: Dict[str, Any] = {
        "model": model,
        "X": X,
        "Y": Y,
        "demos": demos,
        "input_format": [
            "action_type_id",
            "object_id",
            "target_id",
            "has_explicit_xy",
        ],
        "output_format": [
            "action_type_id",
            "object_id",
            "pick_x",
            "pick_y",
            "place_x",
            "place_y",
            "target_id",
            "has_explicit_xy",
        ],
        "num_demos": len(demos),
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\nSaved nearest-neighbor BC policy to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
