# liveness.py
import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/home/light/Desktop/face_auth/face-auth-venv/lib/python3.12/site-packages/"
    "face_recognition_models/models/shape_predictor_68_face_landmarks.dat"
)

LEFT_EYE  = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
NOSE_TIP  = 30

EAR_THRESHOLD  = 0.25
EAR_CONSEC     = 2
HEAD_THRESHOLD = 15


def get_ear(landmarks, eye_points):
    coords = np.array([(landmarks.part(p).x, landmarks.part(p).y)
                       for p in eye_points])
    A   = distance.euclidean(coords[1], coords[5])
    B   = distance.euclidean(coords[2], coords[4])
    C   = distance.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C)


def get_nose_x(landmarks):
    return landmarks.part(NOSE_TIP).x


def check_liveness(max_wait_seconds=10, ui_callback=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    blink_count       = 0
    eye_closed_frames = 0
    blink_done        = False
    head_start_x      = None
    head_moved        = False
    start_time        = time.time()

    while True:
        elapsed   = time.time() - start_time
        remaining = max_wait_seconds - elapsed
        if remaining <= 0:
            cap.release()
            return False

        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            landmarks = predictor(gray, faces[0])

            left_ear  = get_ear(landmarks, LEFT_EYE)
            right_ear = get_ear(landmarks, RIGHT_EYE)
            avg_ear   = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= EAR_CONSEC:
                    blink_count += 1
                    if blink_count >= 1:
                        blink_done = True
                eye_closed_frames = 0

            nose_x = get_nose_x(landmarks)
            if head_start_x is None:
                head_start_x = nose_x
            else:
                if abs(nose_x - head_start_x) > HEAD_THRESHOLD:
                    head_moved = True

        if ui_callback:
            ui_callback(blink_done, head_moved, remaining)

        if blink_done and head_moved:
            cap.release()
            return True

    cap.release()
    return False
