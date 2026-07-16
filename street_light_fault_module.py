import time

# =========================
# CONFIGURATION
# =========================
LDR_NIGHT_THRESHOLD = 300

CURRENT_ON_THRESHOLD = 0.2
CURRENT_OFF_THRESHOLD = 0.05

OFF_CONFIRM_TIME = 5          # seconds OFF before bulb fault
TWINKLE_WINDOW = 20           # seconds to monitor transitions
TRANSITION_LIMIT = 10         # number of ON/OFF transitions
TWINKLE_RECOVERY_TIME = 5     # seconds of stable state


# =========================
# STATE VARIABLES
# =========================
previous_state = None
off_start_time = None
transition_timestamps = []
last_fault = None
stable_start_time = None


# =========================
# RESET SYSTEM
# =========================
def reset_all():
    global previous_state
    global off_start_time
    global transition_timestamps
    global last_fault
    global stable_start_time

    previous_state = None
    off_start_time = None
    transition_timestamps.clear()
    last_fault = None
    stable_start_time = None


# =========================
# MAIN FAULT DETECTION
# =========================
def check_street_light_fault(ldr_value, current_value):

    global previous_state
    global off_start_time
    global transition_timestamps
    global last_fault
    global stable_start_time

    current_time = time.time()

    # =========================
    # DAYTIME CHECK
    # =========================
    if ldr_value >= LDR_NIGHT_THRESHOLD:
        reset_all()
        return "DAYTIME"

    # =========================
    # DETERMINE LIGHT STATE
    # =========================
    if current_value > CURRENT_ON_THRESHOLD:
        state = "ON"

    elif current_value < CURRENT_OFF_THRESHOLD:
        state = "OFF"

    else:
        state = previous_state

    if state is None:
        state = "OFF"

    # =========================
    # BULB FAULT DETECTION
    # =========================
    if state == "OFF":

        if off_start_time is None:
            off_start_time = current_time

        elif current_time - off_start_time >= OFF_CONFIRM_TIME:
            last_fault = "BULB_FAULT"

    else:
        off_start_time = None

        if last_fault == "BULB_FAULT":
            last_fault = None

    # =========================
    # TRANSITION TRACKING
    # =========================
    if previous_state is not None and state != previous_state:

        transition_timestamps.append(current_time)

        # reset stability timer
        stable_start_time = None

    previous_state = state

    # =========================
    # REMOVE OLD TRANSITIONS
    # =========================
    transition_timestamps = [
        t for t in transition_timestamps
        if current_time - t <= TWINKLE_WINDOW
    ]

    # =========================
    # TWINKLING DETECTION
    # =========================
    if len(transition_timestamps) >= TRANSITION_LIMIT:
        last_fault = "TWINKLING_FAULT"

    # =========================
    # TWINKLE AUTO RECOVERY
    # =========================
    if last_fault == "TWINKLING_FAULT":

        if stable_start_time is None:
            stable_start_time = current_time

        elif current_time - stable_start_time >= TWINKLE_RECOVERY_TIME:

            last_fault = None
            transition_timestamps.clear()

    # =========================
    # FINAL RESULT
    # =========================
    return last_fault if last_fault else "NORMAL"