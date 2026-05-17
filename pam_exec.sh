#!/bin/bash
# allow root to access display
xhost +local:root 2>/dev/null

export DISPLAY=:1
sleep 1

exec /home/light/Desktop/face_auth/face-auth-venv/bin/python3 /home/light/Desktop/face_auth/face_ui.py
