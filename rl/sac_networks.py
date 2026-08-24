# -*- coding: utf-8 -*-

"""
sac_networks.py

PyTorch neural networks for Soft Actor-Critic (SAC).

Includes:
- Gaussian stochastic actor
- Tanh action squashing
- Twin Q critics

The actor outputs continuous actions in [-1, 1].
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPSILON = 1e-6


# ================================================================
# Weight initialization
# ================================================================

def init_layer(layer: nn.Linear) -> None:
    """
    Xavier initialization for linear layers.
    """

    nn.init.xavier_uniform_(layer.weight)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# ================================================================
# SAC Actor
# ================================================================

class GaussianActor(nn.Module):
    """
    Stochastic Gaussian actor for SAC.

    Input:
        state

    Output:
        Gaussian mean and log standard deviation.

    Sampling:
        state
          ↓
        Normal(mean, std)
          ↓
        reparameterized sample
          ↓
        tanh
          ↓
        action in [-1, 1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:

        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        # ---------------------------------------------------------
        # Shared feature layers
        # ---------------------------------------------------------

        self.fc1 = nn.Linear(
            self.state_dim,
            self.hidden_dim,
        )

        self.fc2 = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        # ---------------------------------------------------------
        # Gaussian outputs
        # ---------------------------------------------------------

        self.mean_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        self.log_std_layer = nn.Linear(
            self.hidden_dim,
            self.action_dim,
        )

        # Initialize
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.mean_layer)
        init_layer(self.log_std_layer)

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        x = F.relu(
            self.fc1(state)
        )

        x = F.relu(
            self.fc2(x)
        )

        mean = self.mean_layer(
            x
        )

        log_std = self.log_std_layer(
            x
        )

        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    # -------------------------------------------------------------
    # Sample action
    # -------------------------------------------------------------

    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns:

            action
            log_probability
            deterministic_action
        """

        mean, log_std = self.forward(
            state
        )

        std = log_std.exp()

        # Gaussian distribution
        normal = torch.distributions.Normal(
            mean,
            std,
        )

        # Reparameterization trick:
        #
        # z = mean + std * epsilon
        #
        # rsample() keeps gradients.
        z = normal.rsample()

        # Squash action into [-1, 1].
        action = torch.tanh(z)

        # ---------------------------------------------------------
        # Log probability
        #
        # SAC requires correction for tanh transformation.
        # ---------------------------------------------------------

        log_prob = normal.log_prob(z)

        log_prob -= torch.log(
            1.0
            - action.pow(2)
            + EPSILON
        )

        log_prob = log_prob.sum(
            dim=-1,
            keepdim=True,
        )

        # Deterministic evaluation action
        deterministic_action = torch.tanh(
            mean
        )

        return (
            action,
            log_prob,
            deterministic_action,
        )


# ================================================================
# Q Critic
# ================================================================

class QNetwork(nn.Module):
    """
    One state-action value network.

    Input:
        state + action

    Output:
        Q(s, a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:

        super().__init__()

        input_dim = (
            int(state_dim)
            + int(action_dim)
        )

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.q = nn.Linear(
            hidden_dim,
            1,
        )

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.q)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat(
            [
                state,
                action,
            ],
            dim=-1,
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        q_value = self.q(
            x
        )

        return q_value


# ================================================================
# Twin Critics
# ================================================================

class TwinQNetwork(nn.Module):
    """
    Two independent Q-functions.

    SAC uses:

        min(Q1, Q2)

    to reduce positive value overestimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:

        super().__init__()

        self.q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        q1 = self.q1(
            state,
            action,
        )

        q2 = self.q2(
            state,
            action,
        )

        return q1, q2


# ================================================================
# Standalone test
# ================================================================

def main() -> None:

    print(
        "\n--- SAC Networks Test ---\n"
    )

    device = torch.device(
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
    batch_size = 32

    # -------------------------------------------------------------
    # Networks
    # -------------------------------------------------------------

    actor = GaussianActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(device)

    critics = TwinQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(device)

    # -------------------------------------------------------------
    # Fake state batch
    # -------------------------------------------------------------

    states = torch.randn(
        batch_size,
        state_dim,
        device=device,
    )

    # -------------------------------------------------------------
    # Actor
    # -------------------------------------------------------------

    (
        actions,
        log_probs,
        deterministic_actions,
    ) = actor.sample(
        states
    )

    print(
        "\nActor outputs:"
    )

    print(
        "actions:",
        actions.shape,
    )

    print(
        "log_probs:",
        log_probs.shape,
    )

    print(
        "deterministic_actions:",
        deterministic_actions.shape,
    )

    print(
        "\nAction range:"
    )

    print(
        "min:",
        float(actions.min().detach())
    )

    print(
        "max:",
        float(actions.max().detach())
    )

    # -------------------------------------------------------------
    # Critics
    # -------------------------------------------------------------

    q1, q2 = critics(
        states,
        actions,
    )

    print(
        "\nCritic outputs:"
    )

    print(
        "Q1:",
        q1.shape,
    )

    print(
        "Q2:",
        q2.shape,
    )

    # -------------------------------------------------------------
    # Gradient test
    # -------------------------------------------------------------

    test_loss = (
        q1.mean()
        + q2.mean()
        - log_probs.mean()
    )

    test_loss.backward()

    actor_grad_exists = any(
        parameter.grad is not None
        for parameter
        in actor.parameters()
    )

    critic_grad_exists = any(
        parameter.grad is not None
        for parameter
        in critics.parameters()
    )

    print(
        "\nGradient test:"
    )

    print(
        "Actor gradients:",
        actor_grad_exists,
    )

    print(
        "Critic gradients:",
        critic_grad_exists,
    )

    # -------------------------------------------------------------
    # Parameter counts
    # -------------------------------------------------------------

    actor_parameters = sum(
        p.numel()
        for p in actor.parameters()
    )

    critic_parameters = sum(
        p.numel()
        for p in critics.parameters()
    )

    print(
        "\nParameter counts:"
    )

    print(
        "Actor:",
        actor_parameters,
    )

    print(
        "Twin critics:",
        critic_parameters,
    )

    print(
        "\nSAC network test completed successfully."
    )


if __name__ == "__main__":
    main()
