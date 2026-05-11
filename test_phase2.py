from language.parser import parse_command
from grounding.object_grounder import ground_object

commands = [
    "pick the red cube and place it at x 0.55 y -0.25",
    "pick the blue cube",
    "move the yellow block to x 0.45 y 0.20",
    "pick the green box and place it at x 0.60 y 0.15",
    "pick the box",
]

for cmd in commands:
    parsed = parse_command(cmd)
    body_name = ground_object(parsed)

    print("\nCommand:", cmd)
    print("Parsed:", parsed)
    print("Grounded MuJoCo object:", body_name)
