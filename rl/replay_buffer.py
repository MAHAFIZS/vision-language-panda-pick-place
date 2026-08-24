# -*- coding: utf-8 -*-

"""
replay_buffer.py

Replay buffer for Soft Actor-Critic (SAC).

Stores transitions:

    state
    action
    reward
    next_state
    done

and returns random mini-batches as PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Parameters
    ----------
    state_dim:
        Dimension of observation/state vector.

    action_dim:
        Dimension of action vector.

    capacity:
        Maximum number of transitions stored.

    device:
        PyTorch device used when returning mini-batches.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)

        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)

        # ---------------------------------------------------------
        # Pre-allocated memory
        # ---------------------------------------------------------

        self.states = np.zeros(
            (self.capacity, self.state_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (self.capacity, self.action_dim),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (self.capacity, self.state_dim),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        # Current insertion index.
        self.ptr = 0

        # Number of valid transitions stored.
        self.size = 0

    # ============================================================
    # Add transition
    # ============================================================

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"State shape must be {(self.state_dim,)}, "
                f"got {state.shape}"
            )

        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Action shape must be {(self.action_dim,)}, "
                f"got {action.shape}"
            )

        if next_state.shape != (self.state_dim,):
            raise ValueError(
                f"Next-state shape must be {(self.state_dim,)}, "
                f"got {next_state.shape}"
            )

        # ---------------------------------------------------------
        # Store transition
        # ---------------------------------------------------------

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = float(done)

        # Circular pointer.
        self.ptr = (
            self.ptr + 1
        ) % self.capacity

        # Size grows until capacity.
        self.size = min(
            self.size + 1,
            self.capacity,
        )

    # ============================================================
    # Sample mini-batch
    # ============================================================

    def sample(
        self,
        batch_size: int,
    ) -> ReplayBatch:

        batch_size = int(batch_size)

        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer. "
                f"Current size={self.size}, "
                f"requested batch_size={batch_size}"
            )

        indices = self.rng.integers(
            low=0,
            high=self.size,
            size=batch_size,
        )

        states = torch.as_tensor(
            self.states[indices],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            self.actions[indices],
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            self.rewards[indices],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            self.next_states[indices],
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            self.dones[indices],
            dtype=torch.float32,
            device=self.device,
        )

        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

    # ============================================================
    # Helpers
    # ============================================================

    def __len__(self) -> int:
        return self.size

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:
        return self.size >= int(batch_size)


# ================================================================
# Standalone test
# ================================================================

def main() -> None:

    print(
        "\n--- Replay Buffer Test ---\n"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    state_dim = 23
    action_dim = 3

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=1000,
        device=device,
        seed=42,
    )

    rng = np.random.default_rng(123)

    # -------------------------------------------------------------
    # Add fake transitions
    # -------------------------------------------------------------

    for i in range(100):

        state = rng.normal(
            size=state_dim,
        ).astype(np.float32)

        action = rng.uniform(
            -1.0,
            1.0,
            size=action_dim,
        ).astype(np.float32)

        reward = float(
            rng.normal()
        )

        next_state = (
            state
            + 0.01
            * rng.normal(
                size=state_dim
            ).astype(np.float32)
        )

        done = bool(
            i % 20 == 19
        )

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    print(
        "Stored transitions:",
        len(buffer),
    )

    # -------------------------------------------------------------
    # Sample batch
    # -------------------------------------------------------------

    batch = buffer.sample(
        batch_size=32
    )

    print(
        "\nBatch shapes:"
    )

    print(
        "states:",
        batch.states.shape,
    )

    print(
        "actions:",
        batch.actions.shape,
    )

    print(
        "rewards:",
        batch.rewards.shape,
    )

    print(
        "next_states:",
        batch.next_states.shape,
    )

    print(
        "dones:",
        batch.dones.shape,
    )

    print(
        "\nTensor device:",
        batch.states.device,
    )

    print(
        "\nReplay buffer test completed successfully."
    )


if __name__ == "__main__":
    main()
