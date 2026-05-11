# -*- coding: utf-8 -*-
"""
sequence_encoder.py

Converts a symbolic pick-and-place command into a robot action sequence.

This is a lightweight VLA-style action planner:
Language command -> parsed task -> symbolic action -> low-level action sequence
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TARGET_POSITIONS = {
    "bin_center": [0.55, 0.25, 0.03],
    "zone_left": [0.55, -0.25, 0.03],
    "zone_right": [0.75, -0.25, 0.03],
}


def build_pick_place_sequence(
    parsed: Dict[str, Any],
    object_positions: Dict[str, List[float]],
    hover_height: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Convert parsed command into a sequence of robot actions.
    """

    obj = parsed.get("obj", "box")
    task = parsed.get("task", "unknown")

    if obj not in object_positions:
        raise ValueError(f"Object '{obj}' not found in object_positions.")

    pick_pos = object_positions[obj]
    pick_xy = [float(pick_pos[0]), float(pick_pos[1])]

    if task == "pick_place_xy":
        place_xy = [float(parsed["x"]), float(parsed["y"])]
        target_name: Optional[str] = None

    elif task == "pick_place":
        target_name = parsed.get("target", "bin_center")
        if target_name not in TARGET_POSITIONS:
            raise ValueError(f"Target '{target_name}' not found in TARGET_POSITIONS.")
        place_xy = TARGET_POSITIONS[target_name][:2]

    else:
        raise ValueError(f"Unsupported task for sequence encoding: {task}")

    sequence = [
        {
            "step": 1,
            "name": "APPROACH_OBJECT",
            "control_mode": "cartesian",
            "target_xy": pick_xy,
            "target_z": pick_pos[2] + hover_height,
            "gripper": "open",
        },
        {
            "step": 2,
            "name": "OPEN_GRIPPER",
            "control_mode": "gripper",
            "gripper": "open",
        },
        {
            "step": 3,
            "name": "DESCEND_TO_OBJECT",
            "control_mode": "cartesian",
            "target_xy": pick_xy,
            "target_z": pick_pos[2] + 0.005,
            "gripper": "open",
        },
        {
            "step": 4,
            "name": "CLOSE_GRIPPER",
            "control_mode": "gripper",
            "gripper": "close",
        },
        {
            "step": 5,
            "name": "LIFT_OBJECT",
            "control_mode": "cartesian",
            "target_xy": pick_xy,
            "target_z": pick_pos[2] + hover_height,
            "gripper": "close",
        },
        {
            "step": 6,
            "name": "MOVE_TO_TARGET",
            "control_mode": "cartesian",
            "target_xy": place_xy,
            "target_z": pick_pos[2] + hover_height,
            "gripper": "close",
            "target_name": target_name,
        },
        {
            "step": 7,
            "name": "DESCEND_TO_PLACE",
            "control_mode": "cartesian",
            "target_xy": place_xy,
            "target_z": 0.03,
            "gripper": "close",
            "target_name": target_name,
        },
        {
            "step": 8,
            "name": "OPEN_GRIPPER",
            "control_mode": "gripper",
            "gripper": "open",
        },
        {
            "step": 9,
            "name": "RETREAT",
            "control_mode": "cartesian",
            "target_xy": place_xy,
            "target_z": 0.03 + hover_height,
            "gripper": "open",
        },
    ]

    return sequence


def print_action_sequence(sequence: List[Dict[str, Any]]) -> None:
    print("\n[ACTION SEQUENCE]")
    for item in sequence:
        print(
            f"{item['step']:02d}. {item['name']} | "
            f"mode={item['control_mode']} | "
            f"xy={item.get('target_xy')} | "
            f"z={item.get('target_z')} | "
            f"gripper={item.get('gripper')}"
        )
