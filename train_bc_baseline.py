# -*- coding: utf-8 -*-
"""
train_bc_baseline.py

Phase 8:
Tiny behavior cloning baseline for the VLA pick-and-place dataset.

Input:
    X = [action_type_id, object_id, target_id, has_explicit_xy]

Target:
    Y = [
        action_type_id,
        object_id,
        pick_x,
        pick_y,
        place_x,
        place_y,
        target_id,
        has_explicit_xy
    ]

This is intentionally simple. It demonstrates how logged VLA demonstrations
can be converted into a trainable action-prediction model.

Run:
    python3 train_bc_baseline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

from dataset.dataset_loader import build_behavior_cloning_arrays


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "bc_policy.pkl"


def train_model(X: np.ndarray, Y: np.ndarray) -> MultiOutputRegressor:
    """
    Train a tiny multi-output regression model.

    RandomForest is used because it works well with very small tabular datasets
    and does not require feature scaling.
    """
    base_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=4,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X, Y)

    return model


def evaluate_model(model: MultiOutputRegressor, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate model on the same tiny dataset.

    For now this is a sanity check, not a real generalization benchmark.
    """
    Y_pred = model.predict(X)

    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)

    return mae, mse


def print_predictions(model: MultiOutputRegressor, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Print predicted vs true action vectors.
    """
    Y_pred = model.predict(X)

    print("\n--- Behavior Cloning Predictions ---")

    for i, (x, y_true, y_pred) in enumerate(zip(X, Y, Y_pred), start=1):
        print(f"\nDemo {i}")
        print("X input:")
        print(x)

        print("Y true:")
        print(np.round(y_true, 4))

        print("Y predicted:")
        print(np.round(y_pred, 4))

        print("absolute error:")
        print(np.round(np.abs(y_true - y_pred), 4))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 8: Tiny Behavior Cloning Baseline ---")

    X, Y, demos = build_behavior_cloning_arrays("datasets")

    print(f"Loaded demos: {len(demos)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    model = train_model(X, Y)

    mae, mse = evaluate_model(model, X, Y)

    print("\n--- Training-set sanity check ---")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

    print_predictions(model, X, Y)

    joblib.dump(
        {
            "model": model,
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
            "mae": mae,
            "mse": mse,
        },
        MODEL_PATH,
    )

    print(f"\nSaved behavior cloning model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
