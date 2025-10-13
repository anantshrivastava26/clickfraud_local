# click_generator.py
import requests
import time
import random
import argparse
import uuid
from datetime import datetime, timezone

API = "http://localhost:8000/register_click"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def gen_human_event(user_id=None):
    if user_id is None:
        user_id = str(uuid.uuid4())
    ev = {
        "user_id": user_id,
        "ad_id": f"site_{random.randint(1,10)}",
        "timestamp": now_iso(),
        "user_agent": "Mozilla/5.0 (Windows NT) HumanDemo",
        "is_bot_simulated": False,
        "impressions": 1,
        "clicks": random.choice([0,1]),
        "device_type": random.choice([0,1,2]),
        "device_conn_type": random.choice([0,1]),
        "device_model": "model_x"
    }
    return ev

def gen_bot_event(user_id=None):
    if user_id is None:
        user_id = "bot-" + str(random.randint(1,20))
    ev = {
        "user_id": user_id,
        "ad_id": f"site_{random.randint(1,200)}",
        "timestamp": now_iso(),
        "user_agent": "bot-script/1.0",
        "is_bot_simulated": True,
        "impressions": 1,
        "clicks": 1,
        "device_type": 0,
        "device_conn_type": 0,
        "device_model": "bot"
    }
    return ev

def run_human(rate=1.0, users=50):
    """rate: events per second per user approx (low)."""
    user_ids = [str(uuid.uuid4()) for _ in range(users)]
    while True:
        uid = random.choice(user_ids)
        ev = gen_human_event(uid)
        try:
            r = requests.post(API, json=ev, timeout=3)
            print("H", uid[:8], r.status_code, r.json())
        except Exception as e:
            print("err", e)
        time.sleep(random.uniform(0.5, 2.0) / rate)

def run_bot(rate=10.0, bot_count=5):
    """rate: events per second per bot (high)."""
    bot_ids = [f"bot-{i}" for i in range(1, bot_count+1)]
    while True:
        uid = random.choice(bot_ids)
        ev = gen_bot_event(uid)
        try:
            r = requests.post(API, json=ev, timeout=3)
            print("B", uid, r.status_code, r.json())
        except Exception as e:
            print("err", e)
        # bots click very fast
        time.sleep(random.uniform(0.05, 0.3) / rate)

def run_mixed(human_rate=1.0, bot_rate=10.0, human_count=30, bot_count=5):
    # alternate human and bots
    import threading
    th = threading.Thread(target=run_human, args=(human_rate, human_count), daemon=True)
    tb = threading.Thread(target=run_bot, args=(bot_rate, bot_count), daemon=True)
    th.start(); tb.start()
    while True:
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human","bot","mixed"], default="mixed")
    parser.add_argument("--human_rate", type=float, default=1.0)
    parser.add_argument("--bot_rate", type=float, default=10.0)
    parser.add_argument("--human_count", type=int, default=30)
    parser.add_argument("--bot_count", type=int, default=5)
    args = parser.parse_args()

    print("Starting generator mode:", args.mode)
    if args.mode == "human":
        run_human(rate=args.human_rate, users=args.human_count)
    elif args.mode == "bot":
        run_bot(rate=args.bot_rate, bot_count=args.bot_count)
    else:
        run_mixed(human_rate=args.human_rate, bot_rate=args.bot_rate,
                  human_count=args.human_count, bot_count=args.bot_count)
