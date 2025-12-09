# history_store_new.py
import json
from typing import List, Dict, Any
from User_files.db import get_conn


def load_user_conversations(username: str) -> List[Dict[str, Any]]:
    """
    Return list of conversations for this user:
    [{ "id": ..., "title": ..., "messages": [...] }, ...]
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, messages
            FROM conversations
            WHERE username = %s
            ORDER BY updated_at DESC
            """,
            (username,),
        )
        rows = cur.fetchall()

    conversations: List[Dict[str, Any]] = []
    for conv_id, title, messages in rows:
        conversations.append(
            {
                "id": conv_id,
                "title": title,
                "messages": messages,  # psycopg will give you a Python list/dict from JSONB
            }
        )
    return conversations


def save_user_conversations(username: str, conversations: List[Dict[str, Any]]) -> None:
    """
    Persist all conversations for this user.
    Simple strategy: upsert each conversation by ID.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        for conv in conversations:
            conv_id = conv["id"]
            title = conv.get("title", "Conversation")
            messages = conv.get("messages", [])

            cur.execute(
                """
                INSERT INTO conversations (id, username, title, messages)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    messages = EXCLUDED.messages,
                    updated_at = NOW()
                """,
                (conv_id, username, title, json.dumps(messages)),
            )

def delete_conversation(username: str, conv_id: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversations WHERE username = %s AND id = %s",
            (username, conv_id)
        )
