import cv2
import time

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if not ret:
    print("❌ Camera not accessible")
    exit()

# =========================
# PARAMETERS
# =========================
MOTION_AREA_THRESHOLD = 2000
MOTION_FRAMES_REQUIRED = 8
ACCIDENT_DELAY = 3   # seconds of no motion after large motion

# =========================
# STATE VARIABLES
# =========================
motion_counter = 0
no_motion_start = None
accident_detected = False

print("🚦 Accident Detection Started")

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    large_motion = False

    # =========================
    # MOTION DETECTION
    # =========================
    for contour in contours:
        if cv2.contourArea(contour) > MOTION_AREA_THRESHOLD:
            large_motion = True

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # =========================
    # LARGE MOTION EVENT
    # =========================
    if large_motion:

        motion_counter += 1
        no_motion_start = None
        accident_detected = False

        cv2.putText(frame1, "Large Motion Detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

    # =========================
    # CHECK FOR NO MOTION
    # =========================
    else:

        if motion_counter >= MOTION_FRAMES_REQUIRED:

            if no_motion_start is None:
                no_motion_start = time.time()

            elif time.time() - no_motion_start > ACCIDENT_DELAY:
                accident_detected = True

        else:
            motion_counter = 0

    # =========================
    # ACCIDENT ALERT
    # =========================
    if accident_detected:

        cv2.putText(frame1,
                    "ACCIDENT DETECTED",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    else:

        cv2.putText(frame1,
                    "Monitoring...",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("Accident Detection", frame1)

    # =========================
    # FRAME UPDATE
    # =========================
    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("System stopped.")