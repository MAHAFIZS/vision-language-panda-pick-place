# -*- coding: utf-8 -*-
"""
dataset_loader.py

Loads saved VLA demonstration JSON files and converts them into
training-ready arrays for behavior cloning / imitation learning.

Current supervised learning format:

Input X:
    [
        action_type_id,
        object_id,
        target_id,
        has_explicit_xy
    ]

Output Y:
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

This is intentionally simple for Phase 7.
Later, X can include language embeddings, image embeddings, or world-state features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_demo_file(path: str | Path) -> Dict[str, Any]:
    """
    Load one demonstration JSON file.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    return record


def load_all_demos(dataset_dir: str | Path = "datasets") -> List[Dict[str, Any]]:
    """
    Load all demo_*.json files from a dataset directory.
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    demo_paths = sorted(dataset_dir.glob("demo_*.json"))

    demos = []
    for path in demo_paths:
        demos.append(load_demo_file(path))

    return demos


def demo_to_training_pair(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one saved demo record into one supervised learning pair.

    X = compact command/object/task features
    Y = action vector
    """
    data = record["data"]

    symbolic = data["symbolic_action"]
    action_vector = data["action_vector"]

    action_type_id = float(symbolic.get("action_type_id", 0))
    object_id = float(symbolic.get("object_id", -1))
    target_id = float(symbolic.get("target_id", 0))

    decoded = data.get("decoded_action", {})
    has_explicit_xy = 1.0 if decoded.get("has_explicit_xy", False) else 0.0

    x = np.array(
        [
            action_type_id,
            object_id,
            target_id,
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    y = np.array(action_vector, dtype=np.float32)

    return x, y


def build_behavior_cloning_arrays(
    dataset_dir: str | Path = "datasets",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Load all demos and build X, Y arrays.

    Returns:
        X: shape [N, 4]
        Y: shape [N, 8]
        demos: original loaded JSON records
    """
    demos = load_all_demos(dataset_dir)

    if not demos:
        raise RuntimeError(f"No demo_*.json files found in {dataset_dir}")

    xs = []
    ys = []

    for record in demos:
        x, y = demo_to_training_pair(record)
        xs.append(x)
        ys.append(y)

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)

    return X, Y, demos


def print_dataset_summary(dataset_dir: str | Path = "datasets") -> None:
    """
    Print a human-readable dataset summary.
    """
    X, Y, demos = build_behavior_cloning_arrays(dataset_dir)

    print("\n--- VLA Demonstration Dataset Summary ---")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Number of demos: {len(demos)}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    print("\nX feature format:")
    print("[action_type_id, object_id, target_id, has_explicit_xy]")

    print("\nY action vector format:")
    print("[action_type_id, object_id, pick_x, pick_y, place_x, place_y, target_id, has_explicit_xy]")

    print("\nFirst X:")
    print(X[0])

    print("\nFirst Y:")
    print(Y[0])

    print("\nLoaded commands:")
    for record in demos:
        demo_id = record.get("demo_id", "unknown")
        command = record["data"].get("language", "")
        print(f"  demo_{int(demo_id):04d}: {command}")


if __name__ == "__main__":
    print_dataset_summary("datasets")
