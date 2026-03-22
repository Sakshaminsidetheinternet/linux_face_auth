import cv2
import face_recognition
import json
import numpy as nu
from pathlib import Path


BASE_DIR = Path("/etc/face-auth")
ENCODINGS = BASE_DIR/ "encoding.json"
SNAPSHOTS = BASE_DIR/ "snapshots"

BASE_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS.mkdir(exist_ok=True)
print("folder ready ")

if ENCODINGS.exists():
    with open(ENCODINGS) as f:
        existing = json.load(f)
    print(f"found{len(existing)} existing ")
else:
    existing = []
    print("no encodings")


total_samples = 5
new_sample = []
attepmts = 1


print(f"\nWe will capture {total_samples} samples of your face.")
print("Good lighting + look directly at camera = better accuracy\n")

while len(new_sample)<total_samples:
    input(f"Sample {len(new_sample)+1}/{total_samples} ")
    cap = cv2.VideoCapture(0)

    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("cam not working")
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    face_encoding  =  face_recognition.face_encodings(rgb_frame)

    if len(face_encoding) == 0 :
        print("no face found ")
        continue
    elif len(face_encoding) > 1:
        print("multiple face detected")
        continue
    new_sample.append(face_encoding[0].tolist())
    print(f"sample {len(new_sample)}")

all_encodings = existing + new_sample

with open(ENCODINGS, 'w') as f:
    json.dump(all_encodings, f)

print(f"\n✅ Done! {len(new_sample)} samples saved to {ENCODINGS}")
