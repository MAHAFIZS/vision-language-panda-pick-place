# -*- coding: utf-8 -*-
"""
train_tiny_diffusion_policy.py

Phase 9:
Tiny diffusion-style action predictor for the VLA pick-and-place dataset.

This is a lightweight denoising diffusion prototype:

Condition:
    X = [action_type_id, object_id, target_id, has_explicit_xy]

Clean action:
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

Training:
    noisy_Y = Y + noise_level * Gaussian noise
    model learns:
        condition + noisy_Y + timestep -> clean Y

This is not a full visual diffusion policy yet.
It is a compact action-space diffusion prototype for robot action vectors.

Run:
    python3 diffusion_policy/train_tiny_diffusion_policy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset_loader import build_behavior_cloning_arrays


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tiny_diffusion_policy.pt"


class TinyDiffusionPolicy(nn.Module):
    """
    Small MLP denoising model.

    Input:
        condition X: 4 dims
        noisy action: 8 dims
        timestep/noise level: 1 dim

    Output:
        denoised action vector: 8 dims
    """

    def __init__(self, condition_dim: int = 4, action_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()

        input_dim = condition_dim + action_dim + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, condition: torch.Tensor, noisy_action: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, noisy_action, noise_level], dim=-1)
        return self.net(x)


def make_noisy_batch(
    clean_action: torch.Tensor,
    min_noise: float = 0.01,
    max_noise: float = 0.40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add random Gaussian noise to clean action vectors.

    Returns:
        noisy_action
        noise_level, shape [batch, 1]
    """
    batch_size = clean_action.shape[0]
    device = clean_action.device

    noise_level = torch.rand(batch_size, 1, device=device) * (max_noise - min_noise) + min_noise
    noise = torch.randn_like(clean_action) * noise_level

    noisy_action = clean_action + noise

    return noisy_action, noise_level


def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 9: Tiny Diffusion Policy Prototype ---")

    X_np, Y_np, demos = build_behavior_cloning_arrays("datasets")

    print(f"Loaded demos: {len(demos)}")
    print(f"X shape: {X_np.shape}")
    print(f"Y shape: {Y_np.shape}")

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

    model = TinyDiffusionPolicy(condition_dim=X.shape[1], action_dim=Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 1000

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for condition, clean_action in loader:
            noisy_action, noise_level = make_noisy_batch(clean_action)

            pred_action = model(condition, noisy_action, noise_level)

            loss = loss_fn(pred_action, clean_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | loss={avg_loss:.6f}")

    # Final sanity check
    with torch.no_grad():
        noisy_Y, noise_level = make_noisy_batch(Y, min_noise=0.25, max_noise=0.25)
        pred_Y = model(X, noisy_Y, noise_level)

    print("\n--- Denoising sanity check ---")
    for i in range(min(5, len(Y))):
        print(f"\nDemo {i + 1}")
        print("Clean action:")
        print(np.round(Y[i].numpy(), 4))
        print("Noisy action:")
        print(np.round(noisy_Y[i].numpy(), 4))
        print("Denoised prediction:")
        print(np.round(pred_Y[i].numpy(), 4))
        print("Absolute error:")
        print(np.round(torch.abs(pred_Y[i] - Y[i]).numpy(), 4))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "condition_dim": X.shape[1],
        "action_dim": Y.shape[1],
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
    }

    torch.save(checkpoint, MODEL_PATH)

    print(f"\nSaved tiny diffusion policy to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
