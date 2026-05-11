# -*- coding: utf-8 -*-
"""
train_preference_ranker.py

Phase 11:
Tiny preference-learning ranker for VLA pick-and-place actions.

The model learns a scalar score:

    score(command_features, action_vector)

Training objective:
    preferred action should score higher than rejected action.

For simplicity, command_features are:
    [action_type_id, object_id, target_id, has_explicit_xy]

Action vector:
    [
        action_type_id,
        object_id,
        pick_x,
        pick_y,
        place_x,
        place_y,
        target_id,
        has_explicit_xy
    ]

Run:
    python3 preference_learning/train_preference_ranker.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


PREFERENCE_PATH = Path("preference_learning/preferences.json")
MODEL_PATH = Path("models/preference_ranker.pkl")


def command_features_from_pair(pair: Dict[str, Any]) -> np.ndarray:
    symbolic = pair["symbolic_action"]

    action_type_id = float(symbolic.get("action_type_id", 0))
    object_id = float(symbolic.get("object_id", 0))
    target_id = float(symbolic.get("target_id", 0))

    parsed = pair["parsed_task"]
    has_explicit_xy = 1.0 if parsed.get("task") == "pick_place_xy" else 0.0

    return np.array(
        [action_type_id, object_id, target_id, has_explicit_xy],
        dtype=np.float32,
    )


def build_ranker_dataset() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if not PREFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Preference dataset not found: {PREFERENCE_PATH}. "
            "Run python3 preference_learning/create_preference_dataset.py first."
        )

    with PREFERENCE_PATH.open("r", encoding="utf-8") as f:
        pref_data = json.load(f)

    pairs = pref_data["pairs"]

    X = []
    y = []

    for pair in pairs:
        command_features = command_features_from_pair(pair)

        preferred_action = np.array(pair["preferred_action_vector"], dtype=np.float32)
        rejected_action = np.array(pair["rejected_action_vector"], dtype=np.float32)

        preferred_input = np.concatenate([command_features, preferred_action], axis=0)
        rejected_input = np.concatenate([command_features, rejected_action], axis=0)

        X.append(preferred_input)
        y.append(1.0)

        X.append(rejected_input)
        y.append(0.0)

    return np.stack(X, axis=0), np.array(y, dtype=np.float32), pairs


def evaluate_pairwise_accuracy(model: RandomForestRegressor, pairs: List[Dict[str, Any]]) -> float:
    correct = 0

    for pair in pairs:
        command_features = command_features_from_pair(pair)

        preferred_action = np.array(pair["preferred_action_vector"], dtype=np.float32)
        rejected_action = np.array(pair["rejected_action_vector"], dtype=np.float32)

        preferred_input = np.concatenate([command_features, preferred_action], axis=0).reshape(1, -1)
        rejected_input = np.concatenate([command_features, rejected_action], axis=0).reshape(1, -1)

        preferred_score = float(model.predict(preferred_input)[0])
        rejected_score = float(model.predict(rejected_input)[0])

        if preferred_score > rejected_score:
            correct += 1

    return correct / max(1, len(pairs))


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 11: Preference Ranker Training ---")

    X, y, pairs = build_ranker_dataset()

    print(f"Preference pairs: {len(pairs)}")
    print(f"Training samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        random_state=42,
    )

    model.fit(X, y)

    y_pred = model.predict(X)
    y_label = (y_pred >= 0.5).astype(float)
    sample_acc = accuracy_score(y, y_label)
    pair_acc = evaluate_pairwise_accuracy(model, pairs)

    print("\n--- Training-set sanity check ---")
    print(f"Sample classification accuracy: {sample_acc:.3f}")
    print(f"Pairwise preference accuracy: {pair_acc:.3f}")

    print("\nExample scores:")
    for i, pair in enumerate(pairs[:5], start=1):
        command_features = command_features_from_pair(pair)

        preferred_action = np.array(pair["preferred_action_vector"], dtype=np.float32)
        rejected_action = np.array(pair["rejected_action_vector"], dtype=np.float32)

        preferred_input = np.concatenate([command_features, preferred_action], axis=0).reshape(1, -1)
        rejected_input = np.concatenate([command_features, rejected_action], axis=0).reshape(1, -1)

        preferred_score = float(model.predict(preferred_input)[0])
        rejected_score = float(model.predict(rejected_input)[0])

        print(f"\nPair {i}")
        print("Command:", pair["command"])
        print(f"Preferred score: {preferred_score:.3f}")
        print(f"Rejected score:  {rejected_score:.3f}")

    bundle = {
        "model": model,
        "input_format": [
            "command_action_type_id",
            "command_object_id",
            "command_target_id",
            "command_has_explicit_xy",
            "candidate_action_type_id",
            "candidate_object_id",
            "candidate_pick_x",
            "candidate_pick_y",
            "candidate_place_x",
            "candidate_place_y",
            "candidate_target_id",
            "candidate_has_explicit_xy",
        ],
        "num_pairs": len(pairs),
        "sample_accuracy": sample_acc,
        "pairwise_accuracy": pair_acc,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"\nSaved preference ranker to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
