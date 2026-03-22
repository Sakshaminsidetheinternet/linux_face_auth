import cv2
import face_recognition
import json
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/etc/face-auth")
ENCODING = BASE_DIR / "encoding.json"
SNAPSHOTS = BASE_DIR / "snapshots"

tolerance = 0.5
max_attempt = 3

def load_encoding():
    if not ENCODING.exists():
        print("no enrolled face found")
        print("run enroll.py first ")
        return None
    with open(ENCODING) as f:
        data = json.load(f)
    return [np.array(e) for e in data]


def capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot oprn camera")
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("no frame")
        return None
    
    return frame

def check_face(frame, known_encoding):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encoding = face_recognition.face_encodings(rgb_frame)

    if len(face_encoding)==0:
        print("no face detected")
        return False, None
    if len(face_encoding) > 1:
        print("more than one face detected")
        return False, None
    
    current_face = face_encoding[0]

    distances = face_recognition.face_distance(known_encoding, current_face)
    
    best_distance = np.min(distances)
    best_index = np.argmin(distances)

    if best_distance <= tolerance:
        return True, best_distance
    else:
        return False, best_distance
    
def save_snapshot(frame):
    SNAPSHOTS.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOTS/ f"failed{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"saved at : {path}")
    return path

def authenticate():
    known_encoding = load_encoding()
    if known_encoding is None:
        exit(1)
    print(f"✓ Loaded {len(known_encoding)} face samples")
    print(f"\nStarting authentication ({max_attempt} attempts)...\n")

    last_frame = None
    for attempt in range(1, max_attempt + 1):
        print(f"Attempt {attempt}/{max_attempt} — look at camera...")

        frame = capture()
        if frame is not None:
            last_frame = frame   # keep last frame for snapshot

        if frame is None:
            continue

        matched, distance = check_face(frame, known_encoding)

        if matched:
            print(f"\n{'='*40}")
            print("✅  ACCESS GRANTED")
            print(f"{'='*40}")
            exit(0)
        else:
            print(f"  ✗ Not recognized\n")

    # all attempts failed
    print(f"\n{'='*40}")
    print("❌  ACCESS DENIED")
    print(f"{'='*40}")

    # save snapshot of last attempt
    if last_frame is not None:
        save_snapshot(last_frame)

    exit(1)

# ── Entry point ───────────────────────────────────────────────────────────────
authenticate()





