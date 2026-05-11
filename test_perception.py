import cv2
import mujoco
import numpy as np

from perception.color_detector import detect_colored_objects


model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)

mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="top_cam")
rgb = renderer.render()

detections = detect_colored_objects(rgb)

print("Detections:")
for d in detections:
    print(d)

# draw detections
img = rgb.copy()

for d in detections:
    u, v = d["center"]
    color = d["color"]
    cv2.circle(img, (u, v), 6, (255, 255, 255), -1)
    cv2.putText(
        img,
        color,
        (u + 8, v),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

# OpenCV expects BGR
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("perception_debug.png", img_bgr)

print("Saved perception_debug.png")
