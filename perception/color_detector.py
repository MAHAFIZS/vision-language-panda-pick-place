import cv2
import numpy as np


COLOR_RANGES = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255)),
    ],
    "green": [
        ((40, 80, 80), (85, 255, 255)),
    ],
    "blue": [
        ((95, 100, 100), (125, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
}


def detect_colored_objects(rgb_image):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    detections = []

    for color, ranges in COLOR_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        kernel = np.ones((3, 3), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask_total,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 300 or area > 3000:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(cnt)

            detections.append({
                "color": color,
                "center": (u, v),
                "bbox": (x, y, w, h),
                "area": float(area),
            })

    return detections