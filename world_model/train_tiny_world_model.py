# -*- coding: utf-8 -*-
"""
train_tiny_world_model.py

Phase 10:
Tiny world model prototype for the VLA pick-and-place dataset.

Goal:
    Predict the next object state after an action.

Input:
    [
        object_id,
        current_x,
        current_y,
        current_z,
        action_type_id,
        pick_x,
        pick_y,
        place_x,
        place_y,
        target_id,
        has_explicit_xy
    ]

Output:
    [
        next_x,
        next_y,
        next_z
    ]

This is a compact world-model prototype:
current state + action -> predicted next state

Run:
    python3 world_model/train_tiny_world_model.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tiny_world_model.pkl"

DATASET_DIR = Path("datasets")


TARGET_POSITIONS = {
    "bin_center": [0.55, 0.25, 0.03],
    "zone_left": [0.55, -0.25, 0.03],
    "zone_right": [0.75, -0.25, 0.03],
}


def load_demo_records(dataset_dir: Path = DATASET_DIR) -> List[Dict[str, Any]]:
    paths = sorted(dataset_dir.glob("demo_*.json"))

    if not paths:
        raise RuntimeError(f"No demo_*.json files found in {dataset_dir}")

    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))

    return records


def infer_next_position(data: Dict[str, Any]) -> List[float]:
    """
    Estimate next object position from demo data.

    For explicit XY command:
        use decoded place_xy

    For named target command:
        use target position from TARGET_POSITIONS
    """
    decoded = data["decoded_action"]
    symbolic = data["symbolic_action"]

    if decoded.get("has_explicit_xy", False):
        place_xy = decoded.get("place_xy")
        if place_xy is None:
            raise ValueError("Explicit XY action has no place_xy.")
        return [float(place_xy[0]), float(place_xy[1]), 0.03]

    target_name = symbolic.get("target_name")
    if target_name in TARGET_POSITIONS:
        return TARGET_POSITIONS[target_name]

    # Fallback: object remains where it was
    object_pos = data["object_grounding"]["object_position"]
    return [
        float(object_pos[0]),
        float(object_pos[1]),
        float(object_pos[2]),
    ]


def record_to_world_model_pair(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one demo record into:

    X = current state + action
    Y = next object position
    """
    data = record["data"]

    object_grounding = data["object_grounding"]
    symbolic = data["symbolic_action"]
    action_vector = data["action_vector"]

    current_pos = object_grounding["object_position"]

    if current_pos is None:
        raise ValueError("Object position is missing.")

    object_id = float(symbolic.get("object_id", -1))

    action_type_id = float(action_vector[0])
    pick_x = float(action_vector[2])
    pick_y = float(action_vector[3])
    place_x = float(action_vector[4])
    place_y = float(action_vector[5])
    target_id = float(action_vector[6])
    has_explicit_xy = float(action_vector[7])

    x = np.array(
        [
            object_id,
            float(current_pos[0]),
            float(current_pos[1]),
            float(current_pos[2]),
            action_type_id,
            pick_x,
            pick_y,
            place_x,
            place_y,
            target_id,
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    y = np.array(infer_next_position(data), dtype=np.float32)

    return x, y


def build_world_model_dataset(dataset_dir: Path = DATASET_DIR) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    records = load_demo_records(dataset_dir)

    xs = []
    ys = []
    kept_records = []

    for record in records:
        try:
            x, y = record_to_world_model_pair(record)
            xs.append(x)
            ys.append(y)
            kept_records.append(record)
        except Exception as e:
            print(f"[WARNING] Skipping demo {record.get('demo_id')}: {e}")

    if not xs:
        raise RuntimeError("No valid world-model training samples found.")

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)

    return X, Y, kept_records


def train_world_model(X: np.ndarray, Y: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42,
    )
    model.fit(X, Y)
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 10: Tiny World Model Prototype ---")

    X, Y, records = build_world_model_dataset(DATASET_DIR)

    print(f"Loaded demos: {len(records)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    print("\nInput format:")
    print([
        "object_id",
        "current_x",
        "current_y",
        "current_z",
        "action_type_id",
        "pick_x",
        "pick_y",
        "place_x",
        "place_y",
        "target_id",
        "has_explicit_xy",
    ])

    print("\nOutput format:")
    print(["next_x", "next_y", "next_z"])

    model = train_world_model(X, Y)
    Y_pred = model.predict(X)

    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)

    print("\n--- Training-set sanity check ---")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")

    for i, (record, y_true, y_pred) in enumerate(zip(records, Y, Y_pred), start=1):
        command = record["data"]["language"]

        print(f"\nDemo {i}")
        print("Command:")
        print(command)

        print("True next object position:")
        print(np.round(y_true, 4))

        print("Predicted next object position:")
        print(np.round(y_pred, 4))

        print("Absolute error:")
        print(np.round(np.abs(y_true - y_pred), 4))

    bundle = {
        "model": model,
        "input_format": [
            "object_id",
            "current_x",
            "current_y",
            "current_z",
            "action_type_id",
            "pick_x",
            "pick_y",
            "place_x",
            "place_y",
            "target_id",
            "has_explicit_xy",
        ],
        "output_format": ["next_x", "next_y", "next_z"],
        "num_demos": len(records),
        "mae": mae,
        "mse": mse,
        "target_positions": TARGET_POSITIONS,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\nSaved tiny world model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
