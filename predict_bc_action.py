# -*- coding: utf-8 -*-
"""
predict_bc_action.py

Loads the tiny behavior cloning model and predicts an action vector
from a new language command.

Run:
    python3 predict_bc_action.py

Example:
    pick the yellow cube and place it at x 0.45 y 0.20
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    decode_action_vector,
)


MODEL_PATH = Path("models/bc_policy.pkl")


def command_to_x(command: str) -> np.ndarray:
    """
    Convert language command into model input X.
    """
    parsed = parse_command(command)
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
            f"Model not found: {MODEL_PATH}. Run python3 train_bc_baseline.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    x = command_to_x(command)
    y_pred = model.predict(x.reshape(1, -1))[0]

    decoded = decode_action_vector(y_pred)

    result = {
        "command": command,
        "model_input": x.tolist(),
        "predicted_action_vector": y_pred.tolist(),
        "decoded_prediction": decoded,
    }

    return result


def main() -> None:
    print("\n--- BC Action Prediction Demo ---")
    print("Type a command, or 'quit' to exit.\n")

    while True:
        command = input("BC> ").strip()

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

            print("\n[PREDICTED ACTION VECTOR]")
            print(np.round(result["predicted_action_vector"], 4))

            print("\n[DECODED PREDICTION]")
            print(result["decoded_prediction"])

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()
