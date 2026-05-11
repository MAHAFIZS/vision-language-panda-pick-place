from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np


# ---------------------------------------------------------
# Object and action vocabularies
# ---------------------------------------------------------

OBJECT_ID_MAP = {
    "box": 0,
    "red_box": 1,
    "green_box": 2,
    "blue_box": 3,
    "yellow_box": 4,
}

ACTION_TYPE_ID_MAP = {
    "unknown": 0,
    "pick_place": 1,
    "pick_place_xy": 2,
    "stack": 3,
    "place_on": 4,
    "sort_all": 5,
    "tower": 6,
}

TARGET_ID_MAP = {
    "none": 0,
    "bin_center": 1,
    "zone_left": 2,
    "zone_right": 3,
}


@dataclass
class SymbolicAction:
    """
    Human-readable action representation.

    This is useful for debugging, logging, README examples,
    and explaining how language is converted into robot action tokens.
    """
    action_type: str
    action_type_id: int
    object_name: str
    object_id: int
    target_name: Optional[str]
    target_id: int
    target_xy: Optional[List[float]]
    control_mode: str
    gripper_sequence: List[str]


def object_name_to_id(object_name: str) -> int:
    return OBJECT_ID_MAP.get(object_name, -1)


def action_type_to_id(action_type: str) -> int:
    return ACTION_TYPE_ID_MAP.get(action_type, 0)


def target_name_to_id(target_name: Optional[str]) -> int:
    if target_name is None:
        return TARGET_ID_MAP["none"]
    return TARGET_ID_MAP.get(target_name, TARGET_ID_MAP["none"])


def encode_symbolic_action(parsed_command: Dict[str, Any]) -> SymbolicAction:
    """
    Convert a parsed language command into a symbolic action token.

    Example input:
        {
            "task": "pick_place_xy",
            "obj": "red_box",
            "x": 0.55,
            "y": -0.45
        }

    Example output:
        SymbolicAction(
            action_type="pick_place_xy",
            action_type_id=2,
            object_name="red_box",
            object_id=1,
            target_name=None,
            target_id=0,
            target_xy=[0.55, -0.45],
            control_mode="cartesian",
            gripper_sequence=["open", "close", "open"]
        )
    """

    task = parsed_command.get("task", "unknown")
    obj = parsed_command.get("obj", "box")
    target = parsed_command.get("target", None)

    target_xy = None
    if "x" in parsed_command and "y" in parsed_command:
        target_xy = [
            float(parsed_command["x"]),
            float(parsed_command["y"]),
        ]

    symbolic = SymbolicAction(
        action_type=task,
        action_type_id=action_type_to_id(task),
        object_name=obj,
        object_id=object_name_to_id(obj),
        target_name=target,
        target_id=target_name_to_id(target),
        target_xy=target_xy,
        control_mode="cartesian",
        gripper_sequence=["open", "close", "open"],
    )

    return symbolic


def symbolic_action_to_dict(action: SymbolicAction) -> Dict[str, Any]:
    return asdict(action)


def encode_action_vector(
    parsed_command: Dict[str, Any],
    object_positions: Optional[Dict[str, List[float]]] = None,
) -> np.ndarray:
    """
    Convert a parsed command into a numerical action vector.

    Vector format:
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

    If object_positions is given:
        object_positions = {
            "red_box": [0.40, -0.30, 0.03],
            "green_box": [0.50, -0.30, 0.03],
        }

    If object position is unknown, pick_x and pick_y are set to 0.
    """

    symbolic = encode_symbolic_action(parsed_command)

    pick_x = 0.0
    pick_y = 0.0

    if object_positions is not None and symbolic.object_name in object_positions:
        pos = object_positions[symbolic.object_name]
        pick_x = float(pos[0])
        pick_y = float(pos[1])

    place_x = 0.0
    place_y = 0.0
    has_explicit_xy = 0.0

    if symbolic.target_xy is not None:
        place_x = float(symbolic.target_xy[0])
        place_y = float(symbolic.target_xy[1])
        has_explicit_xy = 1.0

    vector = np.array(
        [
            float(symbolic.action_type_id),
            float(symbolic.object_id),
            pick_x,
            pick_y,
            place_x,
            place_y,
            float(symbolic.target_id),
            has_explicit_xy,
        ],
        dtype=np.float32,
    )

    return vector


def decode_action_vector(vector: np.ndarray) -> Dict[str, Any]:
    """
    Decode a numerical action vector back into a readable dictionary.

    This is useful for debugging and later policy-learning experiments.
    """

    action_id_to_name = {v: k for k, v in ACTION_TYPE_ID_MAP.items()}
    object_id_to_name = {v: k for k, v in OBJECT_ID_MAP.items()}
    target_id_to_name = {v: k for k, v in TARGET_ID_MAP.items()}

    action_type_id = int(vector[0])
    object_id = int(vector[1])
    target_id = int(vector[6])
    has_explicit_xy = bool(vector[7] > 0.5)

    decoded = {
        "action_type": action_id_to_name.get(action_type_id, "unknown"),
        "object_name": object_id_to_name.get(object_id, "unknown_object"),
        "pick_xy": [float(vector[2]), float(vector[3])],
        "target_name": target_id_to_name.get(target_id, "none"),
        "has_explicit_xy": has_explicit_xy,
    }

    if has_explicit_xy:
        decoded["place_xy"] = [float(vector[4]), float(vector[5])]
    else:
        decoded["place_xy"] = None

    return decoded
