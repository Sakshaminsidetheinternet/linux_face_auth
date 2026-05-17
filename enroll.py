# enroll.py
import cv2
import face_recognition
import json
import numpy as np
from pathlib import Path

BASE_DIR  = Path("/etc/face-auth")
ENCODINGS = BASE_DIR / "encodings.json"
SNAPSHOTS = BASE_DIR / "snapshots"

BASE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS.mkdir(exist_ok=True)
print("✓ Folders ready")

if ENCODINGS.exists():
    with open(ENCODINGS) as f:
        existing = json.load(f)
    print(f"✓ Found {len(existing)} existing samples")
else:
    existing = []
    print("✓ Starting fresh")

TOTAL_SAMPLES = 5
new_samples   = []

print(f"\nCapturing {TOTAL_SAMPLES} samples — good lighting, look at camera\n")

while len(new_samples) < TOTAL_SAMPLES:
    input(f"Sample {len(new_samples)+1}/{TOTAL_SAMPLES} — press ENTER...")

    cap = cv2.VideoCapture(0)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  ✗ Camera error — try again")
        continue

    rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encoding = face_recognition.face_encodings(rgb_frame)

    if len(face_encoding) == 0:
        print("  ✗ No face detected — try again")
        continue
    elif len(face_encoding) > 1:
        print("  ✗ Multiple faces — only you in frame")
        continue

    new_samples.append(face_encoding[0].tolist())
    print(f"  ✓ Sample {len(new_samples)} captured")

all_encodings = existing + new_samples
with open(ENCODINGS, "w") as f:
    json.dump(all_encodings, f)

print(f"\n✅ Done! {len(new_samples)} samples saved to {ENCODINGS}")
