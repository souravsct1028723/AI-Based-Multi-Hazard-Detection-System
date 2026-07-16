import cv2
import numpy as np
import time

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
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

print("Models Loaded")


# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🚦 Final Smart Street System Started")


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
    # STREET LIGHT SIMULATION
    # =========================
    ldr_value = 100
    current_value = 0.5

    street_fault = check_street_light_fault(
        ldr_value,
        current_value
    )

    if street_fault in ["BULB_FAULT", "TWINKLING_FAULT"]:

        alert_active = True
        alert_type = street_fault
        alert_start_time = time.time()


    # =========================
    # YOLO ACCIDENT DETECTION
    # =========================
    results = accident_model(frame)

    for r in results:

        boxes = r.boxes

        if boxes is not None and len(boxes) > 0:

            alert_active = True
            alert_type = "ACCIDENT"
            alert_start_time = time.time()

            break


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

            if prev_body_center:

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

            alert_active = True
            alert_type = "FALL"
            alert_start_time = time.time()

    else:
        fall_start_time = None


    # =========================
    # ALERT DISPLAY
    # =========================
    display_text = "Monitoring..."
    display_color = (0, 255, 0)

    if alert_active:

        elapsed = time.time() - alert_start_time

        if elapsed < DISPLAY_DURATION:

            if alert_type == "ACCIDENT":
                display_text = "🚨 ROAD ACCIDENT DETECTED"

            elif alert_type == "FALL":
                display_text = "🚨 HUMAN COLLAPSE DETECTED"

            elif alert_type == "BULB_FAULT":
                display_text = "💡 BULB FAULT DETECTED"

            elif alert_type == "TWINKLING_FAULT":
                display_text = "💡 TWINKLING FAULT DETECTED"

            display_color = (0, 0, 255)

        else:
            alert_active = False
            alert_type = None


    cv2.putText(frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                display_color,
                2)


    # =========================
    # STREET LIGHT VISUAL
    # =========================
    bulb_center = (frame.shape[1] - 100, 100)
    bulb_radius = 30

    base_color = (0, 255, 255)

    if street_fault == "BULB_FAULT":
        base_color = (40, 40, 40)

    elif street_fault == "TWINKLING_FAULT":

        if int(time.time() * 4) % 2 == 0:
            base_color = (0, 0, 255)
        else:
            base_color = (40, 40, 40)

    cv2.line(frame,
             (bulb_center[0], bulb_center[1] + bulb_radius),
             (bulb_center[0], bulb_center[1] + 150),
             (80, 80, 80),
             8)

    cv2.circle(frame,
               bulb_center,
               bulb_radius,
               base_color,
               -1)

    cv2.putText(frame,
                "Street Light",
                (frame.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)


    # =========================
    # DISPLAY WINDOW
    # =========================
    cv2.imshow("Final Smart Street System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()

print("System stopped.")