# -*- coding: utf-8 -*-
"""
random_policy_eval.py

Phase 12:
Random policy evaluation using the tiny RL-style reward function.

This script compares:
    1. correct rule-based action
    2. random candidate actions
    3. best random action found

Run:
    python3 rl/random_policy_eval.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rl.reward_function import (
    expected_action_from_command,
    compute_action_reward,
)


COMMANDS = [
    "pick the red cube and place it at x 0.55 y -0.45",
    "move the yellow block to x 0.45 y 0.20",
    "put the green cube to the right",
    "pick the blue cube and place it in the bin",
    "put the red cube to the left",
]


def sample_random_action() -> np.ndarray:
    """
    Sample a random action vector.

    Format:
    [
        action_type_id,
        object_id,
        pick_x,
        pick_y,
        place_x,
        place_y,
        target_id,
        has_explicit_xy
    ]
    """
    action_type_id = random.choice([1.0, 2.0])
    object_id = random.choice([1.0, 2.0, 3.0, 4.0])
    pick_x = random.uniform(0.35, 0.75)
    pick_y = random.uniform(-0.35, -0.20)
    place_x = random.uniform(0.35, 0.80)
    place_y = random.uniform(-0.50, 0.30)
    target_id = random.choice([0.0, 1.0, 2.0, 3.0])
    has_explicit_xy = random.choice([0.0, 1.0])

    return np.array(
        [
            action_type_id,
            object_id,
            pick_x,
            pick_y,
            place_x,
            place_y,
            target_id,
            has_explicit_xy,
        ],
        dtype=np.float32,
    )


def evaluate_random_policy(command: str, num_samples: int = 50) -> Dict[str, Any]:
    """
    Evaluate many random actions and return the best one.
    """
    _, correct_action = expected_action_from_command(command)
    correct_result = compute_action_reward(command, correct_action)

    random_results = []

    for _ in range(num_samples):
        candidate = sample_random_action()
        result = compute_action_reward(command, candidate)
        random_results.append(result)

    random_results.sort(key=lambda r: r["reward"], reverse=True)
    best_random = random_results[0]
    avg_random_reward = float(np.mean([r["reward"] for r in random_results]))

    return {
        "command": command,
        "correct_action_reward": correct_result["reward"],
        "correct_action_vector": correct_result["candidate_action_vector"],
        "best_random_reward": best_random["reward"],
        "best_random_action_vector": best_random["candidate_action_vector"],
        "best_random_decoded": best_random["decoded_candidate"],
        "average_random_reward": avg_random_reward,
        "num_random_samples": num_samples,
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("\n--- Phase 12: Random Policy Evaluation ---")
    print("Tiny RL-style reward evaluator for VLA pick-and-place actions.\n")

    for command in COMMANDS:
        result = evaluate_random_policy(command, num_samples=100)

        print("=" * 80)
        print("[COMMAND]")
        print(result["command"])

        print("\nCorrect action reward:")
        print(round(result["correct_action_reward"], 4))

        print("\nBest random reward:")
        print(round(result["best_random_reward"], 4))

        print("\nAverage random reward:")
        print(round(result["average_random_reward"], 4))

        print("\nBest random decoded action:")
        print(result["best_random_decoded"])

        print("\nCorrect action vector:")
        print(np.round(result["correct_action_vector"], 4))

        print("\nBest random action vector:")
        print(np.round(result["best_random_action_vector"], 4))

    print("\nDone.")


if __name__ == "__main__":
    main()
