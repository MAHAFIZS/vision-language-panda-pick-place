# -*- coding: utf-8 -*-
"""
reward_function.py

Phase 12:
Tiny RL-style reward function for VLA pick-and-place actions.

The reward evaluates whether a candidate action vector matches the intended
language command.

This is not full RL training yet. It is the first RL component:
    state/command + action -> reward

Action vector format:
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

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    encode_action_vector,
    decode_action_vector,
)


OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


TARGET_POSITIONS = {
    "none": None,
    "bin_center": [0.55, 0.25],
    "zone_left": [0.55, -0.25],
    "zone_right": [0.75, -0.25],
}


def expected_action_from_command(command: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Parse command and return the expected rule-based action vector.
    """
    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    expected = encode_action_vector(parsed, OBJECT_POSITIONS).astype(np.float32)
    return parsed, expected


def compute_action_reward(
    command: str,
    candidate_action: np.ndarray,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute reward for one candidate action.

    Reward components:
        +1.0 correct action type
        +1.0 correct object
        +1.0 correct target id or explicit XY mode
        +1.0 close placement position
        - distance penalty for placement error
    """
    parsed, expected = expected_action_from_command(command)

    candidate = np.array(candidate_action, dtype=np.float32).copy()

    # Round discrete action fields.
    candidate[0] = round(float(candidate[0]))  # action_type_id
    candidate[1] = round(float(candidate[1]))  # object_id
    candidate[6] = round(float(candidate[6]))  # target_id
    candidate[7] = 1.0 if float(candidate[7]) >= 0.5 else 0.0

    expected_discrete = expected.copy()
    expected_discrete[0] = round(float(expected_discrete[0]))
    expected_discrete[1] = round(float(expected_discrete[1]))
    expected_discrete[6] = round(float(expected_discrete[6]))
    expected_discrete[7] = 1.0 if float(expected_discrete[7]) >= 0.5 else 0.0

    reward = 0.0
    components: Dict[str, float] = {}

    # 1. Action type
    if candidate[0] == expected_discrete[0]:
        components["action_type_reward"] = 1.0
    else:
        components["action_type_reward"] = -1.0
    reward += components["action_type_reward"]

    # 2. Object
    if candidate[1] == expected_discrete[1]:
        components["object_reward"] = 1.0
    else:
        components["object_reward"] = -1.0
    reward += components["object_reward"]

    # 3. Explicit XY flag
    if candidate[7] == expected_discrete[7]:
        components["mode_reward"] = 0.5
    else:
        components["mode_reward"] = -0.5
    reward += components["mode_reward"]

    # 4. Target/placement
    has_explicit_xy = bool(expected_discrete[7] >= 0.5)

    if has_explicit_xy:
        expected_xy = expected[4:6]
        candidate_xy = candidate[4:6]

        dist = float(np.linalg.norm(candidate_xy - expected_xy))
        placement_reward = max(0.0, 1.0 - dist / 0.30)
        distance_penalty = -dist

        components["placement_distance"] = dist
        components["placement_reward"] = placement_reward
        components["distance_penalty"] = distance_penalty

        reward += placement_reward
        reward += distance_penalty

    else:
        if candidate[6] == expected_discrete[6]:
            components["target_reward"] = 1.0
        else:
            components["target_reward"] = -1.0
        reward += components["target_reward"]

    # 5. Pick location consistency
    expected_pick = expected[2:4]
    candidate_pick = candidate[2:4]
    pick_dist = float(np.linalg.norm(candidate_pick - expected_pick))
    pick_reward = max(0.0, 0.5 - pick_dist)

    components["pick_distance"] = pick_dist
    components["pick_reward"] = pick_reward

    reward += pick_reward

    decoded_candidate = decode_action_vector(candidate)

    result = {
        "command": command,
        "parsed": parsed,
        "expected_action_vector": expected.tolist(),
        "candidate_action_vector": candidate.tolist(),
        "decoded_candidate": decoded_candidate,
        "reward": float(reward),
        "components": components,
    }

    if verbose:
        print("\n[COMMAND]")
        print(command)

        print("\n[EXPECTED ACTION]")
        print(np.round(expected, 4))

        print("\n[CANDIDATE ACTION]")
        print(np.round(candidate, 4))

        print("\n[DECODED CANDIDATE]")
        print(decoded_candidate)

        print("\n[REWARD]")
        print(float(reward))

        print("\n[COMPONENTS]")
        print(components)

    return result


def reward_for_rule_based_action(command: str) -> Dict[str, Any]:
    """
    Convenience function: evaluate the correct rule-based action.
    """
    _, expected = expected_action_from_command(command)
    return compute_action_reward(command, expected, verbose=True)


if __name__ == "__main__":
    reward_for_rule_based_action("put the green cube to the right")
