COLOR_TO_OBJECT = {
    "red": "red_box",
    "green": "green_box",
    "blue": "blue_box",
    "yellow": "yellow_box",
}


def perception_ground_object(parsed_command, detections):
    obj = parsed_command.get("obj", "box")

    for color, body_name in COLOR_TO_OBJECT.items():
        if body_name == obj:
            visible = any(d["color"] == color for d in detections)
            return {
                "obj": body_name,
                "color": color,
                "visible": visible,
            }

    return {
        "obj": obj,
        "color": None,
        "visible": True,
    }
