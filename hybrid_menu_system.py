import os
import cv2
import numpy as np
import logging
import asyncio
import websockets
import json
import time

from ultralytics import YOLO

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from street_light_fault_module import check_street_light_fault

logging.getLogger("absl").setLevel(logging.ERROR)

# =========================
# WEBSOCKET ALERT
# =========================
async def send_alert(alert_data):

    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(alert_data))

        print("🚨 ALERT SENT:", alert_data)

    except:
        print("❌ WebSocket connection failed")


# =========================
# LOAD MODELS
# =========================
print("Loading models...")

# YOLO Accident Model
accident_model = YOLO("best.pt")

# Pose Model
base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

print("Models Loaded Successfully")


# =========================
# ACCIDENT + COLLAPSE TEST
# =========================
def test_camera_detection():

    image_path = input("Enter image file path: ")

    frame = cv2.imread(image_path)

    if frame is None:
        print("❌ Invalid image path")
        return

    print("Image Loaded Successfully")

    # -------------------------
    # YOLO ACCIDENT DETECTION
    # -------------------------
    print("\n===== ACCIDENT DETECTION =====")

    results = accident_model(frame)

    accident_found = False

    for r in results:

        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:

            accident_found = True

            conf = float(boxes.conf[0])

            print("🚨 ACCIDENT DETECTED")
            print("Confidence:", round(conf, 3))

            alert = {
                "type": "ACCIDENT",
                "confidence": conf,
                "location": "Menu Image Input",
                "timestamp": int(time.time() * 1000)
            }

            asyncio.run(send_alert(alert))

            break

    if not accident_found:
        print("NO ACCIDENT")


    # -------------------------
    # HUMAN COLLAPSE DETECTION
    # -------------------------
    print("\n===== HUMAN COLLAPSE DETECTION =====")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = pose_landmarker.detect(mp_image)

    if result.pose_landmarks:

        lm = result.pose_landmarks[0]

        try:

            ls, rs = lm[11], lm[12]
            lh, rh = lm[23], lm[24]

            shoulder_mid_x = (ls.x + rs.x) / 2
            shoulder_mid_y = (ls.y + rs.y) / 2

            hip_mid_x = (lh.x + rh.x) / 2
            hip_mid_y = (lh.y + rh.y) / 2

            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y

            angle = abs(np.degrees(np.arctan2(dy, dx)))

            print("Body Angle:", round(angle, 2))

            if angle < 30 or angle > 150:

                print("🚨 HUMAN COLLAPSE DETECTED")

                alert = {
                    "type": "FALL",
                    "confidence": 1.0,
                    "location": "Menu Image Input",
                    "timestamp": int(time.time() * 1000)
                }

                asyncio.run(send_alert(alert))

            else:
                print("NO HUMAN COLLAPSE")

        except:
            print("Pose detection error")

    else:
        print("No pose detected")


# =========================
# STREET LIGHT TEST
# =========================
def test_street_light():

    try:
        ldr_value = float(input("Enter LDR value: "))
    except:
        print("❌ Invalid LDR input")
        return

    print("\nContinuous Street Light Monitoring Mode")
    print("Enter current values continuously.")
    print("Enter -1 to stop testing.\n")

    while True:

        try:
            current_value = float(input("Enter current value: "))
        except:
            print("❌ Invalid input")
            continue

        if current_value == -1:
            print("Exiting Street Light Test Mode\n")
            break

        result = check_street_light_fault(ldr_value, current_value)

        print("Street Light Status:", result)

        if result in ["BULB_FAULT", "TWINKLING_FAULT"]:

            alert = {
                "type": result,
                "confidence": 1.0,
                "location": "Street Light 1",
                "timestamp": int(time.time() * 1000)
            }

            asyncio.run(send_alert(alert))


# =========================
# MAIN MENU
# =========================
while True:

    print("\n===== HYBRID SMART SYSTEM =====")
    print("1. Camera Detection (Image Input)")
    print("2. Street Light Testing")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        test_camera_detection()

    elif choice == "2":
        test_street_light()

    elif choice == "3":
        print("System Stopped")
        break

    else:
        print("Invalid Choice")