# -*- coding: utf-8 -*-

"""
sac_agent.py

Soft Actor-Critic implementation for the Panda reaching task.

Includes:
- Gaussian actor
- twin critics
- target critics
- automatic entropy tuning
- Bellman target
- actor update
- critic update
- soft target update
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.sac_networks import GaussianActor, TwinQNetwork
from rl.replay_buffer import ReplayBatch


@dataclass
class SACConfig:
    state_dim: int = 23
    action_dim: int = 3

    hidden_dim: int = 256

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    initial_alpha: float = 0.2


class SACAgent:

    def __init__(
        self,
        config: SACConfig,
        device: str | torch.device = "cpu",
    ) -> None:

        self.config = config
        self.device = torch.device(device)

        # =========================================================
        # Actor
        # =========================================================

        self.actor = GaussianActor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # =========================================================
        # Twin critics
        # =========================================================

        self.critics = TwinQNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        # Target critics start as exact copies.
        self.target_critics = copy.deepcopy(
            self.critics
        ).to(self.device)

        # Target networks are never directly optimized.
        for parameter in self.target_critics.parameters():
            parameter.requires_grad = False

        # =========================================================
        # Optimizers
        # =========================================================

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )

        self.critic_optimizer = Adam(
            self.critics.parameters(),
            lr=config.critic_lr,
        )

        # =========================================================
        # Automatic entropy temperature
        # =========================================================

        # Common SAC choice:
        # target_entropy = -action_dimension
        self.target_entropy = -float(
            config.action_dim
        )

        self.log_alpha = torch.tensor(
            np.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        self.alpha_optimizer = Adam(
            [self.log_alpha],
            lr=config.alpha_lr,
        )

        self.update_count = 0

    # =============================================================
    # Alpha
    # =============================================================

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # =============================================================
    # Action selection
    # =============================================================

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (
            self.config.state_dim,
        ):
            raise ValueError(
                f"Expected state shape "
                f"{(self.config.state_dim,)}, "
                f"got {state.shape}"
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        (
            sampled_action,
            _,
            deterministic_action,
        ) = self.actor.sample(
            state_tensor
        )

        if deterministic:
            action = deterministic_action
        else:
            action = sampled_action

        return (
            action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # =============================================================
    # Critic target
    # =============================================================

    def _compute_target_q(
        self,
        batch: ReplayBatch,
    ) -> torch.Tensor:

        with torch.no_grad():

            (
                next_actions,
                next_log_probs,
                _,
            ) = self.actor.sample(
                batch.next_states
            )

            (
                target_q1,
                target_q2,
            ) = self.target_critics(
                batch.next_states,
                next_actions,
            )

            target_min_q = torch.min(
                target_q1,
                target_q2,
            )

            # Entropy-regularized target:
            #
            # r + gamma * (1-done)
            #       * [min(Q1,Q2) - alpha*log pi]
            soft_target_q = (
                target_min_q
                - self.alpha.detach()
                * next_log_probs
            )

            target_q = (
                batch.rewards
                + self.config.gamma
                * (1.0 - batch.dones)
                * soft_target_q
            )

        return target_q

    # =============================================================
    # Critic update
    # =============================================================

    def _update_critics(
        self,
        batch: ReplayBatch,
    ) -> Dict[str, float]:

        target_q = self._compute_target_q(
            batch
        )

        q1, q2 = self.critics(
            batch.states,
            batch.actions,
        )

        q1_loss = F.mse_loss(
            q1,
            target_q,
        )

        q2_loss = F.mse_loss(
            q2,
            target_q,
        )

        critic_loss = (
            q1_loss
            + q2_loss
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critics.parameters(),
            max_norm=10.0,
        )

        self.critic_optimizer.step()

        return {
            "critic_loss": float(
                critic_loss.detach().cpu().item()
            ),
            "q1_loss": float(
                q1_loss.detach().cpu().item()
            ),
            "q2_loss": float(
                q2_loss.detach().cpu().item()
            ),
            "q1_mean": float(
                q1.detach().mean().cpu().item()
            ),
            "q2_mean": float(
                q2.detach().mean().cpu().item()
            ),
            "target_q_mean": float(
                target_q.detach().mean().cpu().item()
            ),
        }

    # =============================================================
    # Actor update
    # =============================================================

    def _update_actor(
        self,
        batch: ReplayBatch,
    ) -> Dict[str, float]:

        (
            actions,
            log_probs,
            _,
        ) = self.actor.sample(
            batch.states
        )

        q1, q2 = self.critics(
            batch.states,
            actions,
        )

        min_q = torch.min(
            q1,
            q2,
        )

        # SAC actor objective:
        #
        # E[alpha * log(pi(a|s)) - Q(s,a)]
        actor_loss = (
            self.alpha.detach()
            * log_probs
            - min_q
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=10.0,
        )

        self.actor_optimizer.step()

        return {
            "actor_loss": float(
                actor_loss.detach().cpu().item()
            ),
            "log_prob_mean": float(
                log_probs.detach().mean().cpu().item()
            ),
            "policy_q_mean": float(
                min_q.detach().mean().cpu().item()
            ),
        }

    # =============================================================
    # Entropy temperature update
    # =============================================================

    def _update_alpha(
        self,
        batch: ReplayBatch,
    ) -> Dict[str, float]:

        with torch.no_grad():

            (
                _,
                log_probs,
                _,
            ) = self.actor.sample(
                batch.states
            )

        alpha_loss = -(
            self.log_alpha
            * (
                log_probs
                + self.target_entropy
            )
        ).mean()

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )

        alpha_loss.backward()

        self.alpha_optimizer.step()

        return {
            "alpha_loss": float(
                alpha_loss.detach().cpu().item()
            ),
            "alpha": float(
                self.alpha.detach().cpu().item()
            ),
            "target_entropy": float(
                self.target_entropy
            ),
        }

    # =============================================================
    # Target-network update
    # =============================================================

    @torch.no_grad()
    def _soft_update_targets(
        self,
    ) -> None:

        tau = self.config.tau

        for (
            target_parameter,
            parameter,
        ) in zip(
            self.target_critics.parameters(),
            self.critics.parameters(),
        ):

            target_parameter.data.mul_(
                1.0 - tau
            )

            target_parameter.data.add_(
                tau * parameter.data
            )

    # =============================================================
    # Complete SAC update
    # =============================================================

    def update(
        self,
        batch: ReplayBatch,
    ) -> Dict[str, float]:

        critic_metrics = (
            self._update_critics(
                batch
            )
        )

        actor_metrics = (
            self._update_actor(
                batch
            )
        )

        alpha_metrics = (
            self._update_alpha(
                batch
            )
        )

        self._soft_update_targets()

        self.update_count += 1

        return {
            **critic_metrics,
            **actor_metrics,
            **alpha_metrics,
            "update_count": self.update_count,
        }

    # =============================================================
    # Save
    # =============================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "config": self.config.__dict__,
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_count": self.update_count,
        }

        torch.save(
            checkpoint,
            path,
        )

    # =============================================================
    # Load
    # =============================================================

    def load(
        self,
        path: str | Path,
    ) -> None:

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critics.load_state_dict(
            checkpoint["critics"]
        )

        self.target_critics.load_state_dict(
            checkpoint[
                "target_critics"
            ]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint[
                "actor_optimizer"
            ]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint[
                "critic_optimizer"
            ]
        )

        self.alpha_optimizer.load_state_dict(
            checkpoint[
                "alpha_optimizer"
            ]
        )

        loaded_log_alpha = checkpoint[
            "log_alpha"
        ].to(self.device)

        with torch.no_grad():
            self.log_alpha.copy_(
                loaded_log_alpha
            )

        self.update_count = int(
            checkpoint.get(
                "update_count",
                0,
            )
        )


# ================================================================
# Standalone test
# ================================================================

def main() -> None:

    print(
        "\n--- SAC Agent Test ---\n"
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

    config = SACConfig(
        state_dim=23,
        action_dim=3,
        hidden_dim=256,
    )

    agent = SACAgent(
        config=config,
        device=device,
    )

    # -------------------------------------------------------------
    # Single-state action test
    # -------------------------------------------------------------

    fake_state = np.zeros(
        config.state_dim,
        dtype=np.float32,
    )

    stochastic_action = agent.select_action(
        fake_state,
        deterministic=False,
    )

    deterministic_action = agent.select_action(
        fake_state,
        deterministic=True,
    )

    print(
        "\nStochastic action:",
        np.round(
            stochastic_action,
            4,
        ),
    )

    print(
        "Deterministic action:",
        np.round(
            deterministic_action,
            4,
        ),
    )

    print(
        "Action range valid:",
        bool(
            np.all(
                stochastic_action
                >= -1.0
            )
            and np.all(
                stochastic_action
                <= 1.0
            )
        ),
    )

    # -------------------------------------------------------------
    # Fake replay batch
    # -------------------------------------------------------------

    batch_size = 64

    from rl.replay_buffer import ReplayBatch

    states = torch.randn(
        batch_size,
        config.state_dim,
        device=device,
    )

    actions = torch.rand(
        batch_size,
        config.action_dim,
        device=device,
    ) * 2.0 - 1.0

    rewards = torch.randn(
        batch_size,
        1,
        device=device,
    )

    next_states = torch.randn(
        batch_size,
        config.state_dim,
        device=device,
    )

    dones = torch.randint(
        low=0,
        high=2,
        size=(batch_size, 1),
        device=device,
    ).float()

    batch = ReplayBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    # -------------------------------------------------------------
    # Training update
    # -------------------------------------------------------------

    metrics = agent.update(
        batch
    )

    print(
        "\nUpdate metrics:"
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:20s}: "
                f"{value:+.6f}"
            )

        else:
            print(
                f"{key:20s}: "
                f"{value}"
            )

    print(
        "\nCurrent alpha:",
        agent.alpha.item(),
    )

    print(
        "Update count:",
        agent.update_count,
    )

    print(
        "\nSAC agent test completed successfully."
    )


if __name__ == "__main__":
    main()
