# notify.py
import requests
import json
import os
from pathlib import Path
from datetime import datetime

# ── Load credentials from config file ────────────────────────────────────────
CONFIG_FILE = Path("/home/light/Desktop/face_auth/config.json")

def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)

config    = load_config()
BOT_TOKEN = config["bot_token"]
CHAT_ID   = config["chat_id"]

# ── Paths ─────────────────────────────────────────────────────────────────────
ATTEMPT_FILE = Path("/tmp/face-auth-attempts.json")


# ── Failure counter ───────────────────────────────────────────────────────────
def get_failure_count():
    try:
        if ATTEMPT_FILE.exists():
            with open(ATTEMPT_FILE) as f:
                data = json.load(f)
            return data.get("count", 0)
    except:
        pass
    return 0


def increment_count():
    try:
        count = get_failure_count() + 1
        with open(ATTEMPT_FILE, "w") as f:
            json.dump({"count": count}, f)
        return count
    except:
        return 1


def reset_count():
    try:
        with open(ATTEMPT_FILE, "w") as f:
            json.dump({"count": 0}, f)
    except:
        pass


# ── Build message ─────────────────────────────────────────────────────────────
def build_message(count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname  = os.uname().nodename
    return (
        f"🚨 *Face Authentication FAILED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🖥  Host:           `{hostname}`\n"
        f"🕐  Time:           `{timestamp}`\n"
        f"🔢  Total failures: `{count}`\n"
        f"📸  Snapshot:       attached below"
    )


# ── Send alert ────────────────────────────────────────────────────────────────
def send_alert(snapshot_path=None):
    count    = increment_count()
    message  = build_message(count)
    base_url = f"https://api.telegram.org/bot{BOT_TOKEN}"

    try:
        if snapshot_path and Path(snapshot_path).exists():
            with open(snapshot_path, "rb") as photo:
                response = requests.post(
                    f"{base_url}/sendPhoto",
                    data={
                        "chat_id":    CHAT_ID,
                        "caption":    message,
                        "parse_mode": "Markdown"
                    },
                    files={"photo": photo},
                    timeout=10
                )
        else:
            response = requests.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id":    CHAT_ID,
                    "text":       message,
                    "parse_mode": "Markdown"
                },
                timeout=10
            )

        if response.ok:
            print(f"✅ Telegram alert sent (failure #{count})")
        else:
            print(f"❌ Telegram error: {response.text}")

    except Exception as e:
        print(f"❌ Could not send alert: {e}")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sending test alert...")
    send_alert(snapshot_path=None)
