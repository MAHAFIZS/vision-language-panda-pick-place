# -*- coding: utf-8 -*-
"""
run_vla_action_demo.py

Safe VLA-style runner for the Panda pick-and-place project.

This file demonstrates:

Language command
→ natural language parser
→ object/action grounding
→ symbolic action encoding
→ numerical action vector
→ optional robot execution through Demo from pickandplace.py

Run:
    python3 run_vla_action_demo.py

Example commands:
    pick the red cube and place it at x 0.55 y -0.45
    move the yellow block to x 0.45 y 0.20
    put the green cube to the right
    pick the blue cube and place it in the bin
"""

from __future__ import annotations

from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    symbolic_action_to_dict,
    encode_action_vector,
    decode_action_vector,
)

# Static known MuJoCo object positions from world.xml
# Later this can be replaced by live perception / MuJoCo state.
OBJECT_POSITIONS = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


def print_vla_pipeline(command: str) -> dict:
    """
    Convert a natural-language command into a structured VLA-style action.
    """
    print("\n" + "=" * 80)
    print("[LANGUAGE COMMAND]")
    print(command)

    parsed = parse_command(command)
    print("\n[PARSED COMMAND]")
    print(parsed)

    task = parsed.get("task", "unknown")
    if task == "unknown":
        print("\n[ERROR] Could not parse command.")
        return parsed

    obj = parsed.get("obj", "box")
    if obj not in OBJECT_POSITIONS:
        print(f"\n[WARNING] Object '{obj}' not found in known object map.")
    else:
        print("\n[OBJECT GROUNDING]")
        print({
            "object_name": obj,
            "object_position": OBJECT_POSITIONS[obj],
            "visible_or_known": True,
        })

    symbolic = encode_symbolic_action(parsed)
    symbolic_dict = symbolic_action_to_dict(symbolic)

    vector = encode_action_vector(parsed, OBJECT_POSITIONS)
    decoded = decode_action_vector(vector)

    print("\n[SYMBOLIC ACTION]")
    print(symbolic_dict)

    print("\n[ACTION VECTOR]")
    print(vector)

    print("\n[DECODED ACTION VECTOR]")
    print(decoded)

    print("\n[VLA SUMMARY]")
    print({
        "language": command,
        "parsed_task": parsed,
        "symbolic_action": symbolic_dict,
        "action_vector": vector.tolist(),
        "decoded_action": decoded,
    })

    return parsed


def main() -> None:
    print("\n--- Vision-Language-Action Pick-and-Place Demo ---")
    print("Type a command, or type 'quit' to exit.\n")
    print("Examples:")
    print("  pick the red cube and place it at x 0.55 y -0.45")
    print("  move the yellow block to x 0.45 y 0.20")
    print("  put the green cube to the right")
    print("  pick the blue cube and place it in the bin\n")

    while True:
        command = input("VLA> ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            print("Exiting VLA demo.")
            break

        if not command:
            continue

        print_vla_pipeline(command)


if __name__ == "__main__":
    main()
