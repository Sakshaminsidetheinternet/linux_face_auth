# face_ui.py
# GUI face authentication with live camera feed

import cv2
import face_recognition
import numpy as np
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/light/Desktop/face_auth")
import liveness
import notify

BASE_DIR  = Path("/etc/face-auth")
ENCODINGS = BASE_DIR / "encodings.json"
TOLERANCE    = 0.5
MAX_ATTEMPTS = 3

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
BLACK      = (15,  15,  15)
WHITE      = (240, 240, 240)
GRAY       = (80,  80,  80)
LIGHT_GRAY = (160, 160, 160)
GREEN      = (80,  200, 80)
RED        = (80,  80,  220)
CYAN       = (200, 200, 80)

W, H = 640, 540   # window size


def load_encodings():
    if not ENCODINGS.exists():
        return None
    with open(ENCODINGS) as f:
        data = json.load(f)
    return [np.array(e) for e in data]


def save_snapshot(frame):
    snap_dir = BASE_DIR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = snap_dir / f"failed_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def draw_text(img, text, pos, color=WHITE, scale=0.6, thickness=1):
    cv2.putText(img, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_centered_text(img, text, y, color=WHITE, scale=0.6, thickness=1):
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x    = (W - size[0]) // 2
    draw_text(img, text, (x, y), color, scale, thickness)


def draw_progress_bar(img, progress, y, color=CYAN):
    bar_x = 80
    bar_w = W - 160
    bar_h = 6
    cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h), GRAY, -1)
    fill = int(bar_w * progress)
    if fill > 0:
        cv2.rectangle(img, (bar_x, y), (bar_x + fill, y + bar_h), color, -1)


def draw_face_box(img, cam_frame, face_found):
    """Draw live camera feed with face detection box."""
    feed_w, feed_h = 320, 240
    feed_x = (W - feed_w) // 2
    feed_y = 60

    resized = cv2.resize(cam_frame, (feed_w, feed_h))
    img[feed_y:feed_y+feed_h, feed_x:feed_x+feed_w] = resized

    # border
    border_color = GREEN if face_found else LIGHT_GRAY
    cv2.rectangle(img,
        (feed_x - 2, feed_y - 2),
        (feed_x + feed_w + 2, feed_y + feed_h + 2),
        border_color, 2)

    # corner decorations
    corner = 20
    thick  = 2
    for cx, cy, dx, dy in [
        (feed_x, feed_y, 1, 1),
        (feed_x + feed_w, feed_y, -1, 1),
        (feed_x, feed_y + feed_h, 1, -1),
        (feed_x + feed_w, feed_y + feed_h, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx*corner, cy), GREEN, thick)
        cv2.line(img, (cx, cy), (cx, cy + dy*corner), GREEN, thick)

    return feed_y + feed_h


def draw_checkmark(img, cx, cy, radius, color):
    p1 = (cx - radius//3, cy)
    p2 = (cx - radius//8, cy + radius//4)
    p3 = (cx + radius//3, cy - radius//4)
    cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
    cv2.line(img, p2, p3, color, 3, cv2.LINE_AA)


def draw_xmark(img, cx, cy, radius, color):
    o = radius // 3
    cv2.line(img, (cx-o, cy-o), (cx+o, cy+o), color, 3, cv2.LINE_AA)
    cv2.line(img, (cx+o, cy-o), (cx-o, cy+o), color, 3, cv2.LINE_AA)


# ── Phase screens ─────────────────────────────────────────────────────────────
def run_liveness_phase(cap):
    """Phase 1 — liveness check with live camera feed."""
    import dlib
    from scipy.spatial import distance as dist

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "/home/light/Desktop/face_auth/face-auth-venv/lib/python3.12/"
        "site-packages/face_recognition_models/models/"
        "shape_predictor_68_face_landmarks.dat"
    )

    LEFT_EYE  = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))
    NOSE_TIP  = 30
    EAR_THRESH = 0.25
    EAR_CONSEC = 2
    HEAD_THRESH = 15

    blink_done        = False
    head_moved        = False
    eye_closed_frames = 0
    blink_count       = 0
    head_start_x      = None
    start_time        = time.time()
    WIN               = "Face Authentication"

    while True:
        elapsed   = time.time() - start_time
        remaining = 15 - elapsed
        if remaining <= 0:
            return False

        ret, frame = cap.read()
        if not ret:
            continue

        # detect face landmarks
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_found = len(faces) > 0

        if face_found:
            lm = predictor(gray, faces[0])

            # blink
            def ear(points):
                c = np.array([(lm.part(p).x, lm.part(p).y) for p in points])
                A = dist.euclidean(c[1], c[5])
                B = dist.euclidean(c[2], c[4])
                C = dist.euclidean(c[0], c[3])
                return (A + B) / (2.0 * C)

            avg_ear = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0
            if avg_ear < EAR_THRESH:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= EAR_CONSEC:
                    blink_count += 1
                    if blink_count >= 1:
                        blink_done = True
                eye_closed_frames = 0

            # head movement
            nose_x = lm.part(NOSE_TIP).x
            if head_start_x is None:
                head_start_x = nose_x
            elif abs(nose_x - head_start_x) > HEAD_THRESH:
                head_moved = True

        # ── Draw UI ───────────────────────────────────────────────────────────
        img = np.full((H, W, 3), BLACK, dtype=np.uint8)

        # title
        draw_centered_text(img, "LIVENESS CHECK", 35,
                           color=WHITE, scale=0.7, thickness=2)
        draw_centered_text(img, "Step 1 of 3", 55, color=GRAY, scale=0.4)

        # camera feed
        bottom = draw_face_box(img, frame, face_found)

        # status items
        y = bottom + 30
        b_color = GREEN if blink_done else LIGHT_GRAY
        h_color = GREEN if head_moved else LIGHT_GRAY
        b_text  = "Blink eyes:      DONE" if blink_done else "Blink eyes:      waiting..."
        h_text  = "Move head:       DONE" if head_moved else "Move head:       waiting..."

        draw_text(img, b_text, (80, y),      color=b_color, scale=0.55)
        draw_text(img, h_text, (80, y + 28), color=h_color, scale=0.55)

        # timer
        draw_text(img, f"Time: {remaining:.1f}s",
                  (80, y + 60), color=GRAY, scale=0.45)

        cv2.imshow(WIN, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        if blink_done and head_moved:
            time.sleep(0.3)
            return True

    return False


def run_scan_phase(cap, known_encodings):
    """Phase 2 — face scan with live camera feed."""
    WIN        = "Face Authentication"
    last_frame = None
    success    = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        tick     = 0
        duration = 50
        captured = False

        while tick < duration or not captured:
            ret, frame = cap.read()
            if ret:
                last_frame = frame

            progress = min(tick / duration, 1.0)

            # build UI
            img = np.full((H, W, 3), BLACK, dtype=np.uint8)

            draw_centered_text(img, "FACE SCAN", 35,
                               color=WHITE, scale=0.7, thickness=2)
            draw_centered_text(img, "Step 2 of 3", 55,
                               color=GRAY, scale=0.4)

            # camera feed
            if last_frame is not None:
                bottom = draw_face_box(img, last_frame, True)
            else:
                bottom = 310

            # progress bar
            draw_progress_bar(img, progress, bottom + 20)

            # attempt dots
            dot_y = bottom + 50
            for i in range(MAX_ATTEMPTS):
                dot_x = W//2 - (MAX_ATTEMPTS-1)*20 + i*40
                color = CYAN if i == attempt-1 else (GRAY if i > attempt-1 else LIGHT_GRAY)
                cv2.circle(img, (dot_x, dot_y), 5, color, -1, cv2.LINE_AA)

            draw_centered_text(img, f"Attempt {attempt} of {MAX_ATTEMPTS}",
                               dot_y + 30, color=LIGHT_GRAY, scale=0.45)
            draw_centered_text(img, "Look directly at camera",
                               dot_y + 55, color=GRAY, scale=0.4)

            cv2.imshow(WIN, img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                return False, None

            tick += 1

            # capture at 60% progress
            if tick == int(duration * 0.6) and not captured:
                captured = True
                if last_frame is not None:
                    rgb       = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                    face_encs = face_recognition.face_encodings(rgb)
                    if face_encs:
                        distances = face_recognition.face_distance(
                            known_encodings, face_encs[0])
                        if float(np.min(distances)) <= TOLERANCE:
                            success = True

        if success:
            break

    return success, last_frame


def show_result(success):
    """Phase 3 — result animation."""
    WIN = "Face Authentication"
    cx  = W // 2
    cy  = H // 2

    for i in range(30):
        img = np.full((H, W, 3), BLACK, dtype=np.uint8)

        if success:
            radius = 40 + i * 2
            color  = GREEN
            draw_centered_text(img, "FACE RECOGNITION", 35,
                               color=WHITE, scale=0.7, thickness=2)
            draw_centered_text(img, "Step 3 of 3", 55,
                               color=GRAY, scale=0.4)
            cv2.circle(img, (cx, cy), radius, color, 2, cv2.LINE_AA)
            draw_checkmark(img, cx, cy, radius, color)
            draw_centered_text(img, "ACCESS GRANTED",
                               cy + radius + 40,
                               color=GREEN, scale=0.8, thickness=2)
            draw_centered_text(img, "Welcome back!",
                               cy + radius + 70,
                               color=LIGHT_GRAY, scale=0.5)
        else:
            color = RED
            draw_centered_text(img, "FACE RECOGNITION", 35,
                               color=WHITE, scale=0.7, thickness=2)
            draw_centered_text(img, "Step 3 of 3", 55,
                               color=GRAY, scale=0.4)
            cv2.circle(img, (cx, cy), 80, color, 2, cv2.LINE_AA)
            draw_xmark(img, cx, cy, 80, color)
            draw_centered_text(img, "ACCESS DENIED",
                               cy + 120,
                               color=RED, scale=0.8, thickness=2)
            draw_centered_text(img, "Falling back to password...",
                               cy + 150,
                               color=LIGHT_GRAY, scale=0.45)

        cv2.imshow(WIN, img)
        cv2.waitKey(40)

    time.sleep(0.5 if success else 1.0)
    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────
def authenticate():
    known_encodings = load_encodings()
    if known_encodings is None:
        sys.exit(1)

    WIN = "Face Authentication"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    # Phase 1 — liveness
    is_live = run_liveness_phase(cap)
    if not is_live:
        show_result(False)
        try:
            notify.send_alert(snapshot_path=None)
        except:
            pass
        sys.exit(1)

    # Phase 2 — face scan
    success, last_frame = run_scan_phase(cap, known_encodings)
    cap.release()

    # Phase 3 — result
    show_result(success)

    if success:
        try:
            notify.reset_count()
        except:
            pass
        sys.exit(0)
    else:
        try:
            if last_frame is not None:
                snap = save_snapshot(last_frame)
                notify.send_alert(snapshot_path=snap)
            else:
                notify.send_alert(snapshot_path=None)
        except:
            pass
        sys.exit(1)


authenticate()
