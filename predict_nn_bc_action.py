# -*- coding: utf-8 -*-
"""
predict_nn_bc_action.py

Nearest-neighbor behavior cloning prediction for VLA actions.

This retrieves the closest saved demonstration and reuses its action vector.

Run:
    python3 predict_nn_bc_action.py

Example commands:
    pick the yellow cube and place it at x 0.45 y 0.20
    put the green cube to the right
    pick the blue cube and place it in the bin
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    decode_action_vector,
)


MODEL_PATH = Path("models/nn_bc_policy.pkl")


def command_to_x(command: str) -> np.ndarray:
    """
    Convert language command into nearest-neighbor input X.

    X format:
        [action_type_id, object_id, target_id, has_explicit_xy]
    """
    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    symbolic = encode_symbolic_action(parsed)

    has_explicit_xy = 1.0 if parsed.get("task") == "pick_place_xy" else 0.0

    x = np.array(
        [
            float(symbolic.action_type_id),
            float(symbolic.object_id),
            float(symbolic.target_id),
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    return x


def predict_action(command: str) -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run python3 train_nn_bc_baseline.py first."
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    Y = bundle["Y"]
    demos = bundle["demos"]

    x = command_to_x(command)

    distances, indices = model.kneighbors(x.reshape(1, -1))
    retrieved_index = int(indices[0, 0])
    distance = float(distances[0, 0])

    y_pred = Y[retrieved_index]
    decoded = decode_action_vector(y_pred)

    retrieved_demo = demos[retrieved_index]
    retrieved_command = retrieved_demo["data"]["language"]

    result = {
        "command": command,
        "model_input": x.tolist(),
        "retrieved_demo_index": retrieved_index,
        "retrieved_demo_id": retrieved_demo.get("demo_id"),
        "retrieved_command": retrieved_command,
        "retrieval_distance": distance,
        "predicted_action_vector": y_pred.tolist(),
        "decoded_prediction": decoded,
    }

    return result


def main() -> None:
    print("\n--- Nearest-Neighbor BC Action Prediction Demo ---")
    print("Type a command, or 'quit' to exit.\n")

    while True:
        command = input("NN-BC> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not command:
            continue

        try:
            result = predict_action(command)

            print("\n[COMMAND]")
            print(result["command"])

            print("\n[MODEL INPUT X]")
            print(result["model_input"])

            print("\n[RETRIEVED DEMONSTRATION]")
            print({
                "demo_id": result["retrieved_demo_id"],
                "index": result["retrieved_demo_index"],
                "command": result["retrieved_command"],
                "distance": result["retrieval_distance"],
            })

            print("\n[PREDICTED ACTION VECTOR]")
            print(np.round(result["predicted_action_vector"], 4))

            print("\n[DECODED PREDICTION]")
            print(result["decoded_prediction"])

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()
