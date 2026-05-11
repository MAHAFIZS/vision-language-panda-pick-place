# -*- coding: utf-8 -*-
"""
predict_tiny_diffusion_action.py

Loads the tiny diffusion-style policy and predicts/denoises an action vector
for a natural-language command.

Because this is a tiny action-space diffusion prototype, we:
1. parse the command
2. build condition X
3. create a noisy initial action
4. repeatedly denoise it
5. post-process discrete action fields before decoding

Run:
    python3 diffusion_policy/predict_tiny_diffusion_action.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    encode_action_vector,
    decode_action_vector,
)


MODEL_PATH = Path("models/tiny_diffusion_policy.pt")


OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


class TinyDiffusionPolicy(nn.Module):
    """
    Same model architecture used in train_tiny_diffusion_policy.py.

    Input:
        condition X: 4 dims
        noisy action: 8 dims
        noise level: 1 dim

    Output:
        denoised action vector: 8 dims
    """

    def __init__(
        self,
        condition_dim: int = 4,
        action_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        input_dim = condition_dim + action_dim + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        condition: torch.Tensor,
        noisy_action: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([condition, noisy_action, noise_level], dim=-1)
        return self.net(x)


def command_to_condition(command: str) -> tuple[dict, np.ndarray]:
    """
    Convert language command into diffusion condition vector.

    Condition format:
        [action_type_id, object_id, target_id, has_explicit_xy]
    """
    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    symbolic = encode_symbolic_action(parsed)

    has_explicit_xy = 1.0 if parsed.get("task") == "pick_place_xy" else 0.0

    condition = np.array(
        [
            float(symbolic.action_type_id),
            float(symbolic.object_id),
            float(symbolic.target_id),
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    return parsed, condition


def load_model() -> TinyDiffusionPolicy:
    """
    Load trained tiny diffusion policy checkpoint.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python3 diffusion_policy/train_tiny_diffusion_policy.py first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = TinyDiffusionPolicy(
        condition_dim=int(checkpoint["condition_dim"]),
        action_dim=int(checkpoint["action_dim"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def postprocess_action_vector(raw_action: np.ndarray) -> np.ndarray:
    """
    Post-process diffusion model output before decoding.

    The model outputs continuous values, but some action-vector fields are
    categorical/discrete:

        index 0 = action_type_id
        index 1 = object_id
        index 6 = target_id
        index 7 = has_explicit_xy flag

    We round or threshold those fields before calling decode_action_vector().
    """
    action = raw_action.copy()

    # Discrete categorical fields
    action[0] = round(float(action[0]))  # action_type_id
    action[1] = round(float(action[1]))  # object_id
    action[6] = round(float(action[6]))  # target_id

    # Binary flag
    action[7] = 1.0 if float(action[7]) >= 0.5 else 0.0

    return action


def diffusion_predict_action(command: str, denoise_steps: int = 10) -> Dict[str, Any]:
    """
    Predict an action vector using the tiny diffusion-style denoising model.
    """
    parsed, condition_np = command_to_condition(command)
    model = load_model()

    condition = torch.tensor(condition_np, dtype=torch.float32).reshape(1, -1)

    # Start from the rule-based action vector plus noise.
    # This keeps the tiny demo stable while still demonstrating denoising.
    base_action_np = encode_action_vector(parsed, OBJECT_POSITIONS).astype(np.float32)
    action = torch.tensor(base_action_np, dtype=torch.float32).reshape(1, -1)

    initial_noise_level = 0.30
    action = action + torch.randn_like(action) * initial_noise_level

    initial_noisy_action = action.clone()

    with torch.no_grad():
        for step in range(denoise_steps, 0, -1):
            noise_value = step / denoise_steps * initial_noise_level
            noise_level = torch.tensor([[noise_value]], dtype=torch.float32)

            action = model(condition, action, noise_level)

    raw_pred_np = action.reshape(-1).numpy()
    clean_pred_np = postprocess_action_vector(raw_pred_np)

    decoded = decode_action_vector(clean_pred_np)

    result = {
        "command": command,
        "parsed": parsed,
        "condition_x": condition_np.tolist(),
        "initial_noisy_action": initial_noisy_action.reshape(-1).numpy().tolist(),
        "raw_denoised_action_vector": raw_pred_np.tolist(),
        "predicted_action_vector": clean_pred_np.tolist(),
        "decoded_prediction": decoded,
    }

    return result


def main() -> None:
    print("\n--- Tiny Diffusion Policy Action Prediction Demo ---")
    print("Type a natural-language robot command, or 'quit' to exit.\n")
    print("Examples:")
    print("  pick the yellow cube and place it at x 0.45 y 0.20")
    print("  put the green cube to the right")
    print("  pick the blue cube and place it in the bin\n")

    while True:
        command = input("DIFFUSION> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not command:
            continue

        try:
            result = diffusion_predict_action(command)

            print("\n[COMMAND]")
            print(result["command"])

            print("\n[PARSED]")
            print(result["parsed"])

            print("\n[CONDITION X]")
            print(result["condition_x"])

            print("\n[INITIAL NOISY ACTION]")
            print(np.round(result["initial_noisy_action"], 4))

            print("\n[RAW DENOISED ACTION]")
            print(np.round(result["raw_denoised_action_vector"], 4))

            print("\n[POST-PROCESSED ACTION VECTOR]")
            print(np.round(result["predicted_action_vector"], 4))

            print("\n[DECODED PREDICTION]")
            print(result["decoded_prediction"])

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()