import bot


def test_chunks_respect_limit_and_keep_lines():
    text = "\n".join(f"row {i} " + "x" * 50 for i in range(300))
    chunks = bot.chunk_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "\n".join(chunks) == text


def test_long_single_line_split():
    assert all(len(c) <= 100 for c in bot.chunk_message("y" * 450, limit=100))


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert bot.send_telegram_alert("hi")["ok"] is False


def test_token_redacted_from_errors(monkeypatch):
    class S:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def post(self, url, **k): raise RuntimeError(f"boom {url}")
    monkeypatch.setattr(bot.requests, "Session", S)
    r = bot.send_telegram_alert("hi", token="SECRET123", chat_id="1")
    assert "SECRET123" not in r["error"]
