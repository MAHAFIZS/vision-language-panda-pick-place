# -*- coding: utf-8 -*-
"""
evaluate_action_reward.py

Interactive reward evaluation demo.

Run:
    python3 rl/evaluate_action_reward.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rl.reward_function import (
    expected_action_from_command,
    compute_action_reward,
)


def main() -> None:
    print("\n--- RL Reward Function Demo ---")
    print("Type a robot command, or 'quit' to exit.\n")

    while True:
        command = input("REWARD> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not command:
            continue

        try:
            _, expected = expected_action_from_command(command)

            print("\nEvaluating correct rule-based action:")
            compute_action_reward(command, expected, verbose=True)

            print("\nEvaluating corrupted wrong-object action:")
            wrong = expected.copy()
            wrong[1] = 1.0 if wrong[1] != 1.0 else 2.0
            compute_action_reward(command, wrong, verbose=True)

            print("\nEvaluating corrupted wrong-placement action:")
            wrong_place = expected.copy()
            wrong_place[4] += 0.20
            wrong_place[5] += 0.20
            compute_action_reward(command, wrong_place, verbose=True)

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()
