# -*- coding: utf-8 -*-
"""
create_preference_dataset.py

Phase 11B:
Stronger preference dataset generator for VLA pick-and-place actions.

Creates multiple preference pairs per demo:

    preferred action > wrong object action
    preferred action > wrong target action
    preferred action > wrong placement action

Output:
    preference_learning/preferences.json

Run:
    python3 preference_learning/create_preference_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DATASET_DIR = Path("datasets")
OUTPUT_PATH = Path("preference_learning/preferences.json")


OBJECT_IDS = {
    "none": 0,
    "red_box": 1,
    "green_box": 2,
    "blue_box": 3,
    "yellow_box": 4,
    "box": 5,
}

TARGET_IDS = {
    "none": 0,
    "bin_center": 1,
    "zone_left": 2,
    "zone_right": 3,
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


def make_wrong_object_action(action_vector: List[float]) -> List[float]:
    bad = list(action_vector)

    true_object_id = int(round(float(bad[1])))

    possible_object_ids = [
        v for v in OBJECT_IDS.values()
        if v not in {0, true_object_id}
    ]

    if possible_object_ids:
        # deterministic alternative for reproducible dataset
        bad[1] = float(possible_object_ids[0])

    return bad


def make_wrong_target_action(action_vector: List[float]) -> List[float]:
    bad = list(action_vector)

    true_target_id = int(round(float(bad[6])))
    has_explicit_xy = int(round(float(bad[7])))

    if has_explicit_xy == 0:
        possible_target_ids = [
            v for v in TARGET_IDS.values()
            if v not in {0, true_target_id}
        ]

        if possible_target_ids:
            bad[6] = float(possible_target_ids[0])
    else:
        # For explicit XY commands, corrupt the XY instead.
        bad[4] = float(bad[4]) + 0.20
        bad[5] = float(bad[5]) - 0.20

    return bad


def make_wrong_position_action(action_vector: List[float]) -> List[float]:
    bad = list(action_vector)

    # Corrupt placement XY.
    bad[4] = float(bad[4]) + 0.20
    bad[5] = float(bad[5]) + 0.20

    return bad


def build_preference_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []

    for record in records:
        data = record["data"]

        command = data["language"]
        parsed_task = data["parsed_task"]
        symbolic_action = data["symbolic_action"]
        preferred_action = data["action_vector"]

        rejected_actions = [
            ("wrong_object", make_wrong_object_action(preferred_action)),
            ("wrong_target", make_wrong_target_action(preferred_action)),
            ("wrong_position", make_wrong_position_action(preferred_action)),
        ]

        for rejection_type, rejected_action in rejected_actions:
            pair = {
                "demo_id": record.get("demo_id"),
                "command": command,
                "parsed_task": parsed_task,
                "symbolic_action": symbolic_action,
                "preferred_action_vector": preferred_action,
                "rejected_action_vector": rejected_action,
                "rejection_type": rejection_type,
                "preference_label": "preferred_over_rejected",
                "reason": f"preferred action matches saved demonstration; rejected action has {rejection_type}",
            }

            pairs.append(pair)

    return pairs


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 11B: Strong Preference Dataset Generation ---")

    records = load_demo_records(DATASET_DIR)
    pairs = build_preference_pairs(records)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Preference pairs for VLA pick-and-place actions",
                "num_pairs": len(pairs),
                "pairs": pairs,
            },
            f,
            indent=2,
        )

    print(f"Loaded demos: {len(records)}")
    print(f"Generated preference pairs: {len(pairs)}")
    print(f"Saved preference dataset to: {OUTPUT_PATH}")

    print("\nExample pair:")
    print(json.dumps(pairs[0], indent=2))


if __name__ == "__main__":
    main()
