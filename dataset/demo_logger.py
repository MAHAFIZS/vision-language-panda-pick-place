# -*- coding: utf-8 -*-
"""
demo_logger.py

Utility for saving VLA-style pick-and-place demonstrations.

Each demonstration contains:
- language command
- parsed task
- object grounding
- symbolic action
- action vector
- decoded action vector
- action sequence

These JSON files can later be used for:
- behavior cloning
- imitation learning
- diffusion policy prototypes
- world model training
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


class DemoLogger:
    def __init__(self, output_dir: str = "datasets") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _next_demo_id(self) -> int:
        existing = sorted(self.output_dir.glob("demo_*.json"))

        if not existing:
            return 1

        ids = []
        for path in existing:
            try:
                stem = path.stem  # demo_0001
                num = int(stem.split("_")[-1])
                ids.append(num)
            except Exception:
                pass

        if not ids:
            return 1

        return max(ids) + 1

    def save_demo(self, demo: Dict[str, Any]) -> Path:
        """
        Save one demonstration as JSON.
        """
        demo_id = self._next_demo_id()

        output_path = self.output_dir / f"demo_{demo_id:04d}.json"

        record = {
            "demo_id": demo_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data": demo,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return output_path
