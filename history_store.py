# history_store.py
import os
import json
from typing import List, Dict, Any

HISTORY_DIR = "history"

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR, exist_ok=True)

def history_path(username: str) -> str:
    safe_user = username.replace("/", "_")
    return os.path.join(HISTORY_DIR, f"{safe_user}.json")

def load_user_conversations(username: str) -> List[Dict[str, Any]]:
    path = history_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt file? Start fresh.
        return []

def save_user_conversations(username: str, conversations: List[Dict[str, Any]]):
    path = history_path(username)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)
