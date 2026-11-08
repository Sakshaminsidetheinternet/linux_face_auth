# Linux Face Authentication Prototype

A Linux-oriented face-authentication experiment built with Python and computer-vision libraries. The repository contains user-enrolment, face matching, blink/head-movement liveness checks, a camera UI and a PAM helper script.

> **Important:** This is an experimental prototype, not a hardened replacement for passwords, hardware security keys or a platform's standard authentication. Do not deploy it as the only way to access a real account or production machine.

## Features in the repository

- **Face enrolment:** captures multiple face samples and stores face encodings.
- **Face matching:** compares a camera frame with enrolled encodings.
- **Liveness challenge:** checks for a blink and head movement using facial landmarks.
- **Camera interface:** displays a live camera workflow and authentication feedback.
- **PAM integration experiment:** includes a shell helper intended to be called from a PAM configuration.

These checks are basic presentation-attack deterrents, not robust anti-spoofing. A blink/head-turn challenge can be defeated by sufficiently capable replay or synthetic attacks.

## Technology

- Python
- OpenCV
- face_recognition
- dlib
- NumPy
- SciPy
- Linux PAM integration (experimental)

## Repository structure

```text
linux_face_auth/
├── enroll.py       # Capture and save face samples
├── face_ui.py      # Camera-based authentication interface
├── liveness.py     # Blink and head-movement challenge
├── notify.py       # Notification helpers
└── pam_exec.sh     # PAM helper script
```

## Setup notes

This project uses native computer-vision dependencies, so installation can vary by Linux distribution and Python version. Create an isolated virtual environment and install compatible versions of OpenCV, face_recognition, dlib, NumPy and SciPy. Check the dependency requirements for the exact versions supported by your system.

The current scripts contain machine-specific absolute paths (for example, paths under `/home/light/Desktop/face_auth` and `/etc/face-auth`). These must be reviewed and adapted before running on another machine.

## Intended workflow

1. Review the source and dependencies.
2. Configure a dedicated local data directory with restrictive permissions.
3. Enrol a test user using `enroll.py`.
4. Run the camera interface and test matching/liveness using a non-critical account.
5. Only experiment with PAM in a disposable VM with console recovery available.

Do not modify system PAM configuration on your primary machine without a tested recovery path. A misconfigured PAM stack can lock you out.

## Privacy and security

Face encodings and snapshots are biometric-related data. Store them with restrictive file permissions, do not commit them to Git, and provide a deletion process. Avoid retaining failed-login snapshots unless there is a clear need and informed consent. Always keep a password-based fallback.

## Demo and screenshots

No verified public demo or screenshots are currently included. A short recording using a test account and synthetic/demo enrolment data would be a suitable visual demonstration.

## Project status

This repository contains a working prototype structure, but its end-to-end behaviour, PAM compatibility and security have not been independently verified. It should be treated as an educational project.

## Author

**Saksham Gupta** — [GitHub](https://github.com/Sakshaminsidetheinternet)
