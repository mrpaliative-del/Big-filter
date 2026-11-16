"""Telegram delivery."""
from __future__ import annotations

import os

import requests

CHUNK_SIZE = 4000  # Telegram's hard limit is 4096 characters


def chunk_message(text: str, limit: int = CHUNK_SIZE) -> list[str]:
    """Split on line boundaries so result rows are never cut in half."""
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > limit:  # pathological single line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def send_telegram_alert(
    message: str,
    parse_mode: str | None = None,
    token: str | None = None,
    chat_id: str | None = None,
) -> dict:
    """Send ``message`` in chunks.

    ``parse_mode`` defaults to plain text: with legacy Markdown a single ``_``
    or ``*`` in a team name makes Telegram reject the whole message.
    Credentials are read at call time, and the token is scrubbed from errors
    (requests exceptions embed the full URL, token included).
    """
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables."}
    if not message or not message.strip():
        return {"ok": False, "error": "Empty message."}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last: dict = {"ok": True}
    with requests.Session() as session:
        for chunk in chunk_message(message):
            payload = {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            try:
                data = session.post(url, json=payload, timeout=10).json()
            except Exception as exc:  # network error or non-JSON reply
                return {"ok": False, "error": str(exc).replace(token, "***")}
            if not data.get("ok"):
                return data
            last = data
    return last
