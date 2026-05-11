import mujoco

from nl_interface import parse_command
from perception.color_detector import detect_colored_objects
from perception.vision_grounder import perception_ground_object


model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="top_cam")
rgb = renderer.render()

detections = detect_colored_objects(rgb)

command = "pick the red cube and place it at x 0.55 y -0.25"
parsed = parse_command(command)
grounded = perception_ground_object(parsed, detections)

print("Command:", command)
print("Parsed:", parsed)
print("Detections:", detections)
print("Vision-grounded object:", grounded)
