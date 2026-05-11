# -*- coding: utf-8 -*-
"""
create_video_action_dataset.py

Creates a simple Video Action Dataset from extracted video frames.

Dataset idea:

    frame image
    + language command
    + action label
    + action phase
    + action sequence

This is a lightweight prototype for Video Action Model / VAM-style work.

Run:
    python3 video_action/create_video_action_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    symbolic_action_to_dict,
    encode_action_vector,
)
from action_sequence.sequence_encoder import build_pick_place_sequence


OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


DEFAULT_VIDEO_NAME = "vla_panda_demo"
DEFAULT_COMMAND = "pick the red cube and place it at x 0.55 y -0.45"

FRAMES_DIR = Path("video_action/frames")
OUTPUT_PATH = Path("video_action/datasets/video_action_dataset.json")


def assign_phase(frame_index: int, total_frames: int) -> str:
    """
    Assign a coarse action phase based on normalized frame progress.

    This is approximate. Later it can be replaced with real timestamps,
    robot state logs, or manual annotations.
    """
    if total_frames <= 1:
        return "UNKNOWN"

    progress = frame_index / max(1, total_frames - 1)

    if progress < 0.12:
        return "APPROACH_OBJECT"
    if progress < 0.22:
        return "OPEN_GRIPPER"
    if progress < 0.35:
        return "DESCEND_TO_OBJECT"
    if progress < 0.45:
        return "CLOSE_GRIPPER"
    if progress < 0.58:
        return "LIFT_OBJECT"
    if progress < 0.72:
        return "MOVE_TO_TARGET"
    if progress < 0.84:
        return "DESCEND_TO_PLACE"
    if progress < 0.92:
        return "OPEN_GRIPPER"
    return "RETREAT"


def build_video_action_dataset(
    video_name: str = DEFAULT_VIDEO_NAME,
    command: str = DEFAULT_COMMAND,
) -> Dict[str, Any]:
    frame_dir = FRAMES_DIR / video_name

    if not frame_dir.exists():
        raise FileNotFoundError(
            f"Frame directory not found: {frame_dir}. "
            "Run extract_video_frames.py first."
        )

    frame_paths = sorted(frame_dir.glob("*.jpg"))

    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")

    parsed = parse_command(command)

    if parsed.get("task") == "unknown":
        raise ValueError(f"Could not parse command: {command}")

    symbolic = encode_symbolic_action(parsed)
    symbolic_dict = symbolic_action_to_dict(symbolic)
    action_vector = encode_action_vector(parsed, OBJECT_POSITIONS).tolist()
    action_sequence = build_pick_place_sequence(parsed, OBJECT_POSITIONS)

    samples: List[Dict[str, Any]] = []

    total_frames = len(frame_paths)

    for i, frame_path in enumerate(frame_paths):
        phase = assign_phase(i, total_frames)

        sample = {
            "sample_id": i + 1,
            "video_name": video_name,
            "frame_path": str(frame_path),
            "frame_index": i,
            "num_frames": total_frames,
            "language_command": command,
            "parsed_task": parsed,
            "symbolic_action": symbolic_dict,
            "action_vector": action_vector,
            "action_phase_label": phase,
            "action_sequence": action_sequence,
        }

        samples.append(sample)

    dataset = {
        "description": "Prototype video-action dataset for VLA Panda pick-and-place",
        "video_name": video_name,
        "language_command": command,
        "num_samples": len(samples),
        "frame_dir": str(frame_dir),
        "samples": samples,
    }

    return dataset


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- Video Action Dataset Builder ---")

    dataset = build_video_action_dataset(
        video_name=DEFAULT_VIDEO_NAME,
        command=DEFAULT_COMMAND,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Video name: {dataset['video_name']}")
    print(f"Command: {dataset['language_command']}")
    print(f"Samples: {dataset['num_samples']}")
    print(f"Saved dataset to: {OUTPUT_PATH}")

    print("\nFirst sample:")
    print(json.dumps(dataset["samples"][0], indent=2))

    print("\nLast sample:")
    print(json.dumps(dataset["samples"][-1], indent=2))


if __name__ == "__main__":
    main()
