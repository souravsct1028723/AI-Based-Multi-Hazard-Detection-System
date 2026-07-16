import cv2
import numpy as np
import time
import asyncio
import json
import websockets

from ultralytics import YOLO

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from street_light_fault_module import check_street_light_fault


# =========================
# CONFIGURATION
# =========================
DISPLAY_DURATION = 10
FALL_CONFIRM_TIME = 2
NO_MOVEMENT_TIME = 2
ALERT_COOLDOWN = 5

CONF_THRESHOLD = 0.85
ACCIDENT_CONFIRM_FRAMES = 8
MIN_BOX_AREA = 8000

last_sent_alert = {}
accident_frame_counter = 0


# =========================
# WEBSOCKET ALERT SYSTEM
# =========================
async def send_alert(message):

    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(message))
    except:
        print("WebSocket connection failed")


def trigger_alert(alert_type, extra_data=None):

    now = time.time()

    if alert_type in last_sent_alert:
        if now - last_sent_alert[alert_type] < ALERT_COOLDOWN:
            return

    last_sent_alert[alert_type] = now

    payload = {
        "type": alert_type,
        "timestamp": int(now * 1000)
    }

    if extra_data:
        payload.update(extra_data)

    print("🚨 ALERT SENT:", payload)

    asyncio.run(send_alert(payload))


# =========================
# LOAD MODELS
# =========================
print("Loading models...")

accident_model = YOLO("best.pt")

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

print("Models loaded")


# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("LIVE HYBRID SYSTEM STARTED")


# =========================
# STATE VARIABLES
# =========================
alert_active = False
alert_type = None
alert_start_time = 0

fall_start_time = None
no_move_start_time = None
prev_body_center = None

frame_timestamp = 0


# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break


    # =========================
    # STREET LIGHT MONITORING
    # =========================
    ldr_value = 100
    current_value = 0.5

    street_fault = check_street_light_fault(
        ldr_value,
        current_value
    )

    if street_fault in ["BULB_FAULT", "TWINKLING_FAULT"]:

        trigger_alert(
            street_fault,
            {"location": "Pole 1"}
        )

        alert_active = True
        alert_type = street_fault
        alert_start_time = time.time()


    # =========================
    # YOLO ACCIDENT DETECTION
    # =========================
    results = accident_model(frame)

    accident_detected_this_frame = False
    detected_severity = None
    detected_conf = 0

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = accident_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1
            area = width * height

            # draw box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

            cv2.putText(frame,
                        f"{class_name} {conf:.2f}",
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2)

            # Ignore tall shapes (likely humans)
            if height > width * 1.2:
                continue

            # Accident detection
            if class_name in ["mild","moderate","severe"]:

                if conf > CONF_THRESHOLD and area > MIN_BOX_AREA:

                    accident_detected_this_frame = True
                    detected_severity = class_name
                    detected_conf = conf


    if accident_detected_this_frame:
        accident_frame_counter += 1
    else:
        accident_frame_counter = 0


    if accident_frame_counter >= ACCIDENT_CONFIRM_FRAMES:

        trigger_alert("ACCIDENT",{
            "confidence": detected_conf,
            "severity": detected_severity,
            "location": "Camera 1"
        })

        alert_active = True
        alert_type = f"ACCIDENT ({detected_severity})"
        alert_start_time = time.time()

        accident_frame_counter = 0


    # =========================
    # HUMAN COLLAPSE DETECTION
    # =========================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = pose_landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1

    lying_down = False
    no_movement = False

    if result.pose_landmarks:

        lm = result.pose_landmarks[0]

        try:

            head = lm[0]
            ls, rs = lm[11], lm[12]
            lh, rh = lm[23], lm[24]
            la, ra = lm[27], lm[28]

            body_height = abs(head.y - max(la.y, ra.y))
            body_width = abs(ls.x - rs.x) + abs(lh.x - rh.x)

            if body_width > body_height:
                lying_down = True

            body_center = (
                (ls.x + rs.x)/2,
                (ls.y + rs.y)/2
            )

            if prev_body_center is not None:

                movement = abs(body_center[0] - prev_body_center[0]) + \
                           abs(body_center[1] - prev_body_center[1])

                if movement < 0.01:

                    if no_move_start_time is None:
                        no_move_start_time = time.time()

                    elif time.time() - no_move_start_time >= NO_MOVEMENT_TIME:
                        no_movement = True

                else:
                    no_move_start_time = None

            prev_body_center = body_center

        except:
            pass


    collapse_detected = lying_down and no_movement

    if collapse_detected:

        if fall_start_time is None:
            fall_start_time = time.time()

        elif time.time() - fall_start_time >= FALL_CONFIRM_TIME:

            trigger_alert(
                "FALL",
                {"location": "Camera 1"}
            )

            alert_active = True
            alert_type = "HUMAN COLLAPSE"
            alert_start_time = time.time()

    else:
        fall_start_time = None


    # =========================
    # DISPLAY ALERT
    # =========================
    display_text = "Monitoring..."
    color = (0,255,0)

    if alert_active:

        elapsed = time.time() - alert_start_time

        if elapsed < DISPLAY_DURATION:

            display_text = alert_type
            color = (0,0,255)

        else:
            alert_active = False
            alert_type = None


    cv2.putText(
        frame,
        display_text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("LIVE HYBRID SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()

print("System stopped")