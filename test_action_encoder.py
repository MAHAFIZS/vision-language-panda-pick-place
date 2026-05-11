from nl_interface import parse_command
from action_encoding.action_encoder import (
    encode_symbolic_action,
    symbolic_action_to_dict,
    encode_action_vector,
    decode_action_vector,
)


object_positions = {
    "red_box": [0.40, -0.30, 0.03],
    "green_box": [0.50, -0.30, 0.03],
    "blue_box": [0.60, -0.30, 0.03],
    "yellow_box": [0.70, -0.30, 0.03],
    "box": [0.40, -0.15, 0.03],
}


commands = [
    "pick the red cube and place it at x 0.55 y -0.45",
    "move the yellow block to x 0.45 y 0.20",
    "put the green cube to the right",
]


for command in commands:
    parsed = parse_command(command)
    symbolic = encode_symbolic_action(parsed)
    vector = encode_action_vector(parsed, object_positions)
    decoded = decode_action_vector(vector)

    print("\nCommand:", command)
    print("Parsed:", parsed)
    print("Symbolic action:", symbolic_action_to_dict(symbolic))
    print("Action vector:", vector)
    print("Decoded vector:", decoded)
