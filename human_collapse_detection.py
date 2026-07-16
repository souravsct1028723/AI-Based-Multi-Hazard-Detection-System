import cv2
import time
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# LOAD POSE MODEL
# =========================
MODEL_PATH = "pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🧍 Human Collapse Detection Started")

# =========================
# PARAMETERS
# =========================
FALL_CONFIRM_TIME = 2
NO_MOVEMENT_TIME = 2
DISPLAY_DURATION = 10

# =========================
# STATE VARIABLES
# =========================
fall_start_time = None
no_move_start_time = None
last_alert_time = 0

prev_head_y = None
prev_body_center = None

frame_timestamp = 0

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

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
    sudden_drop = False
    no_movement = False

    # =========================
    # POSE DETECTION
    # =========================
    if result.pose_landmarks:

        lm = result.pose_landmarks[0]

        try:

            head = lm[0]
            ls, rs = lm[11], lm[12]
            lh, rh = lm[23], lm[24]
            la, ra = lm[27], lm[28]

            # -------------------------
            # BODY GEOMETRY
            # -------------------------
            body_height = abs(head.y - max(la.y, ra.y))
            torso_height = abs(((ls.y + rs.y)/2) - ((lh.y + rh.y)/2))
            body_width = abs(ls.x - rs.x) + abs(lh.x - rh.x)

            # -------------------------
            # LYING POSTURE
            # -------------------------
            if body_width > body_height and torso_height < 0.1:
                lying_down = True

            # -------------------------
            # SUDDEN DROP DETECTION
            # -------------------------
            if prev_head_y is not None:

                drop_speed = head.y - prev_head_y

                if drop_speed > 0.05:
                    sudden_drop = True

            prev_head_y = head.y

            # -------------------------
            # NO MOVEMENT DETECTION
            # -------------------------
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

            # -------------------------
            # DRAW LANDMARKS
            # -------------------------
            for p in lm:
                cx = int(p.x * w)
                cy = int(p.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        except:
            pass

    # =========================
    # COLLAPSE CONFIRMATION
    # =========================
    collapse_detected = lying_down and sudden_drop and no_movement

    if collapse_detected:

        if fall_start_time is None:
            fall_start_time = time.time()

        elif time.time() - fall_start_time >= FALL_CONFIRM_TIME:
            last_alert_time = time.time()

    else:
        fall_start_time = None

    # =========================
    # ALERT DISPLAY
    # =========================
    if time.time() - last_alert_time < DISPLAY_DURATION:

        text = "🚨 HUMAN COLLAPSE DETECTED"
        color = (0, 0, 255)

    else:

        text = "Monitoring..."
        color = (0, 255, 0)

    cv2.putText(frame,
                text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2)

    posture = "LYING" if lying_down else "STANDING"

    cv2.putText(frame,
                f"Posture: {posture}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if lying_down else (0, 255, 0),
                2)

    cv2.imshow("Human Collapse Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()

print("System stopped.")