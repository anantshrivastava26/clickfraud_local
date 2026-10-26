# click_generator.py
"""
Traffic simulator for the demo: fires fake click events at the receiver.

It plays two kinds of visitors against POST /register_click (receiver_fastapi.py):

                 humans                              bots
  user ids       random UUIDs                        "bot-1" .. "bot-N" (a few, reused)
  pacing         ~0.5-2 s between events overall     ~5-30 ms between events overall
  clicks         random 0/1 (about half clicked)     always 1
  ad_id          site_1..site_10                     site_1..site_200 (hops around)
  device_*       random small ints                   always 0

Those differences are what the model picks up: bots produce long, fast sessions with a
click_rate of 1.0 and many distinct ad_ids, so they get flushed by the size cap and
usually come back as outliers. Humans are spread over many users, so each user's events
are far apart; with the default settings the average gap per user (~37 s for 30 users)
exceeds the receiver's 30 s inactivity timeout, so most human "sessions" are only 1-2
events long. That is expected, not a bug.

Usage (receiver must already be running on localhost:8000):
    python click_generator.py                          # mixed humans + bots
    python click_generator.py --mode bot --bot_rate 20
Stop with Ctrl+C.
"""
import requests
import time
import random
import argparse
import uuid
from datetime import datetime, timezone

# Must match the host/port the receiver listens on.
API = "http://localhost:8000/register_click"

def now_iso():
    """UTC timestamp WITH offset ("+00:00"). The receiver needs timezone-aware times;
    a naive timestamp would break its inactivity sweep."""
    return datetime.now(timezone.utc).isoformat()

def gen_human_event(user_id=None):
    """Build one click payload that looks like a casual human visitor."""
    if user_id is None:
        user_id = str(uuid.uuid4())
    ev = {
        "user_id": user_id,
        "ad_id": f"site_{random.randint(1,10)}",
        "timestamp": now_iso(),
        "user_agent": "Mozilla/5.0 (Windows NT) HumanDemo",
        # Ground-truth label, sent for debugging only. The receiver stores it but never
        # feeds it to the model - detection must come from behaviour alone.
        "is_bot_simulated": False,
        "impressions": 1,
        "clicks": random.choice([0,1]),
        "device_type": random.choice([0,1,2]),
        "device_conn_type": random.choice([0,1]),
        "device_model": "model_x"
    }
    return ev

def gen_bot_event(user_id=None):
    """Build one click payload that looks like a click-fraud script: every impression is
    clicked, and the target ad changes constantly."""
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
    """Endless loop of human traffic.

    rate : speed multiplier on the pacing (higher = faster). It scales the single loop's
           delay, i.e. it is the overall event rate, not a per-user rate.
    users: size of the visitor pool; each event comes from a random one of them.
    """
    user_ids = [str(uuid.uuid4()) for _ in range(users)]
    while True:
        uid = random.choice(user_ids)
        ev = gen_human_event(uid)
        # Keep going if the receiver is down/restarting - just log and retry on the
        # next tick instead of crashing the generator.
        try:
            r = requests.post(API, json=ev, timeout=3)
            print("H", uid[:8], r.status_code, r.json())
        except Exception as e:
            print("err", e)
        time.sleep(random.uniform(0.5, 2.0) / rate)

def run_bot(rate=10.0, bot_count=5):
    """Endless loop of bot traffic.

    rate     : speed multiplier on the pacing (higher = faster); overall, not per bot.
    bot_count: number of distinct bot ids. Few ids + fast pacing => each bot accumulates
               events quickly and trips the receiver's per-session event cap.
    """
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
    """Run human and bot traffic at the same time so the dashboard shows both."""
    # alternate human and bots
    import threading
    # Each generator blocks forever, so each gets its own thread. daemon=True so they
    # die automatically when the main thread exits (Ctrl+C).
    th = threading.Thread(target=run_human, args=(human_rate, human_count), daemon=True)
    tb = threading.Thread(target=run_bot, args=(bot_rate, bot_count), daemon=True)
    th.start(); tb.start()
    # Keep the main thread alive (and responsive to Ctrl+C) while the workers run.
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
