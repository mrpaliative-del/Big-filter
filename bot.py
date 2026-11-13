import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    """Pushes automated execution signals headlessly to Telegram monitor."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"Telegram dispatch error: {e}")

if __name__ == "__main__":
    # Example execution run triggered via GitHub Actions background job
    mock_signal = "🚨 *98% Model Signal Alert*\n\n⚽ **Egypt vs Malawi**\n🎯 Markets: `1X & Over 1.5`\n📊 Poisson Prob: `97.4%`"
    
    if TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
        send_telegram_alert(mock_signal)
        print("Headless alert successfully dispatched to Telegram monitor.")
    else:
        print("Bot running in test mode. Configure environment variables for live Telegram dispatch.")
