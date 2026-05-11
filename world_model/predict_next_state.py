# -*- coding: utf-8 -*-
"""
predict_next_state.py

Predict the next object position after a language-conditioned action.

Pipeline:
    language command
    -> parsed command
    -> symbolic/action vector
    -> current object state
    -> world model prediction

Run:
    python3 world_model/predict_next_state.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    encode_action_vector,
)


MODEL_PATH = Path("models/tiny_world_model.pkl")


OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


def build_world_model_input(command: str) -> Dict[str, Any]:
    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    symbolic = encode_symbolic_action(parsed)
    action_vector = encode_action_vector(parsed, OBJECT_POSITIONS)

    obj = parsed.get("obj", "box")

    if obj not in OBJECT_POSITIONS:
        raise ValueError(f"Object '{obj}' not found in OBJECT_POSITIONS.")

    current_pos = OBJECT_POSITIONS[obj]

    x = np.array(
        [
            float(symbolic.object_id),
            float(current_pos[0]),
            float(current_pos[1]),
            float(current_pos[2]),
            float(action_vector[0]),
            float(action_vector[2]),
            float(action_vector[3]),
            float(action_vector[4]),
            float(action_vector[5]),
            float(action_vector[6]),
            float(action_vector[7]),
        ],
        dtype=np.float32,
    )

    return {
        "command": command,
        "parsed": parsed,
        "object_name": obj,
        "current_position": current_pos,
        "symbolic_action": {
            "action_type": symbolic.action_type,
            "action_type_id": symbolic.action_type_id,
            "object_name": symbolic.object_name,
            "object_id": symbolic.object_id,
            "target_name": symbolic.target_name,
            "target_id": symbolic.target_id,
            "target_xy": symbolic.target_xy,
        },
        "action_vector": action_vector.tolist(),
        "world_model_input": x,
    }


def predict_next_state(command: str) -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run python3 world_model/train_tiny_world_model.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    item = build_world_model_input(command)
    x = item["world_model_input"].reshape(1, -1)

    pred_next = model.predict(x)[0]

    result = {
        "command": item["command"],
        "parsed": item["parsed"],
        "object_name": item["object_name"],
        "current_position": item["current_position"],
        "symbolic_action": item["symbolic_action"],
        "action_vector": item["action_vector"],
        "predicted_next_position": pred_next.tolist(),
    }

    return result


def main() -> None:
    print("\n--- Tiny World Model Prediction Demo ---")
    print("Type a natural-language command, or 'quit' to exit.\n")

    print("Examples:")
    print("  pick the red cube and place it at x 0.55 y -0.45")
    print("  put the green cube to the right")
    print("  pick the blue cube and place it in the bin\n")

    while True:
        command = input("WORLD> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not command:
            continue

        try:
            result = predict_next_state(command)

            print("\n[COMMAND]")
            print(result["command"])

            print("\n[PARSED]")
            print(result["parsed"])

            print("\n[OBJECT]")
            print(result["object_name"])

            print("\n[CURRENT POSITION]")
            print(np.round(result["current_position"], 4))

            print("\n[ACTION VECTOR]")
            print(np.round(result["action_vector"], 4))

            print("\n[PREDICTED NEXT OBJECT POSITION]")
            print(np.round(result["predicted_next_position"], 4))

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()
