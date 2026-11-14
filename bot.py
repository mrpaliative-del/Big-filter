import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str) -> dict:
    """Dispatches formatted quantitative signals and live momentum alerts headlessly with chunking support."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {
            "ok": False,
            "error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables.",
        }

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Telegram max message length is 4096 characters. Chunk if necessary.
    max_length = 4000
    messages = [message[i:i + max_length] for i in range(0, len(message), max_length)]
    
    last_response = {"ok": True}
    for chunk in messages:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            res_data = response.json()
            if not res_data.get("ok"):
                return res_data
            last_response = res_data
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
            
    return last_response
