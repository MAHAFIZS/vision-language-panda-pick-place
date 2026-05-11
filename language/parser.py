import re

COLORS = ["red", "blue", "green", "yellow", "gray", "grey"]
OBJECT_WORDS = ["cube", "block", "box"]

def parse_command(command: str):
    command = command.lower().strip()

    result = {
        "action": "pick_place",
        "target_color": None,
        "target_object": None,
        "place_x": None,
        "place_y": None,
        "raw": command,
    }

    for color in COLORS:
        if color in command:
            result["target_color"] = color
            break

    for obj in OBJECT_WORDS:
        if obj in command:
            result["target_object"] = obj
            break

    # Parse commands like: x 0.55 y -0.25
    x_match = re.search(r"x\s*(-?\d+\.?\d*)", command)
    y_match = re.search(r"y\s*(-?\d+\.?\d*)", command)

    if x_match:
        result["place_x"] = float(x_match.group(1))
    if y_match:
        result["place_y"] = float(y_match.group(1))

    # Default placement
    if result["place_x"] is None:
        result["place_x"] = 0.55
    if result["place_y"] is None:
        result["place_y"] = -0.25

    return result
