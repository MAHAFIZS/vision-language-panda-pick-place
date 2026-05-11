# -*- coding: utf-8 -*-
"""
inspect_video_action_dataset.py

Inspect the generated Video Action Dataset.

Run:
    python3 video_action/inspect_video_action_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import json


DATASET_PATH = Path("video_action/datasets/video_action_dataset.json")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Run python3 video_action/create_video_action_dataset.py first."
        )

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]

    print("\n--- Video Action Dataset Summary ---")
    print(f"Video name: {dataset['video_name']}")
    print(f"Command: {dataset['language_command']}")
    print(f"Number of samples: {dataset['num_samples']}")
    print(f"Frame dir: {dataset['frame_dir']}")

    phases = Counter(sample["action_phase_label"] for sample in samples)

    print("\nAction phase counts:")
    for phase, count in phases.items():
        print(f"  {phase}: {count}")

    print("\nExample samples:")

    for idx in [0, len(samples) // 2, len(samples) - 1]:
        sample = samples[idx]
        print("\n" + "-" * 60)
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Frame: {sample['frame_path']}")
        print(f"Phase: {sample['action_phase_label']}")
        print(f"Command: {sample['language_command']}")
        print(f"Action vector: {sample['action_vector']}")


if __name__ == "__main__":
    main()
