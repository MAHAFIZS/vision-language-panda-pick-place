OBJECT_MAP = {
    "red": "red_box",
    "blue": "blue_box",
    "green": "green_box",
    "yellow": "yellow_box",
    "gray": "box",
    "grey": "box",
}

def ground_object(parsed_command):
    color = parsed_command.get("target_color")

    if color is None:
        return "box"

    return OBJECT_MAP.get(color, "box")
