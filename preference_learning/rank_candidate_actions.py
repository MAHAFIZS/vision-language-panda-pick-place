# -*- coding: utf-8 -*-
"""
rank_candidate_actions.py

Ranks candidate robot actions for a language command using the trained
preference ranker.

Run:
    python3 preference_learning/rank_candidate_actions.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    encode_action_vector,
    decode_action_vector,
)


MODEL_PATH = Path("models/preference_ranker.pkl")


OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


def command_features(command: str) -> tuple[dict, np.ndarray]:
    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    symbolic = encode_symbolic_action(parsed)

    has_explicit_xy = 1.0 if parsed.get("task") == "pick_place_xy" else 0.0

    features = np.array(
        [
            float(symbolic.action_type_id),
            float(symbolic.object_id),
            float(symbolic.target_id),
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    return parsed, features


def make_candidate_actions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create one good candidate and several corrupted candidates.
    """
    good = encode_action_vector(parsed, OBJECT_POSITIONS).astype(np.float32)

    candidates = []

    candidates.append({
        "name": "correct_action",
        "action_vector": good,
    })

    # Wrong object candidate
    wrong_object = good.copy()
    wrong_object[1] = 1.0 if good[1] != 1.0 else 2.0
    candidates.append({
        "name": "wrong_object",
        "action_vector": wrong_object,
    })

    # Wrong target candidate
    wrong_target = good.copy()
    wrong_target[6] = 2.0 if good[6] != 2.0 else 3.0
    candidates.append({
        "name": "wrong_target",
        "action_vector": wrong_target,
    })

    # Wrong placement candidate
    wrong_place = good.copy()
    wrong_place[4] = float(wrong_place[4]) + 0.20
    wrong_place[5] = float(wrong_place[5]) + 0.20
    candidates.append({
        "name": "wrong_place_xy",
        "action_vector": wrong_place,
    })

    return candidates


def score_candidate(model: Any, cmd_features: np.ndarray, action_vector: np.ndarray) -> float:
    x = np.concatenate([cmd_features, action_vector], axis=0).reshape(1, -1)
    return float(model.predict(x)[0])


def rank_actions(command: str) -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Preference ranker not found: {MODEL_PATH}. "
            "Run python3 preference_learning/train_preference_ranker.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    parsed, cmd_features = command_features(command)
    candidates = make_candidate_actions(parsed)

    scored = []
    for candidate in candidates:
        action_vector = candidate["action_vector"]
        score = score_candidate(model, cmd_features, action_vector)
        decoded = decode_action_vector(action_vector)

        scored.append({
            "name": candidate["name"],
            "score": score,
            "action_vector": action_vector.tolist(),
            "decoded": decoded,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "command": command,
        "parsed": parsed,
        "ranked_candidates": scored,
    }


def main() -> None:
    print("\n--- Preference-Based Action Ranking Demo ---")
    print("Type a natural-language command, or 'quit' to exit.\n")

    while True:
        command = input("PREF> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not command:
            continue

        try:
            result = rank_actions(command)

            print("\n[COMMAND]")
            print(result["command"])

            print("\n[PARSED]")
            print(result["parsed"])

            print("\n[RANKED CANDIDATE ACTIONS]")
            for i, item in enumerate(result["ranked_candidates"], start=1):
                print(f"\nRank {i}: {item['name']}")
                print(f"Score: {item['score']:.3f}")
                print("Decoded:", item["decoded"])
                print("Action vector:", np.round(item["action_vector"], 4))

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()
