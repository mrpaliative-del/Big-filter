import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402

client = TestClient(app_module.app)
TXT = "ENGLAND: Premier League\n15:00 Liverpool - Chelsea"


def test_scan_ok():
    r = client.post("/scan", json={"raw_text": TXT})
    assert r.status_code == 200 and r.json()["summary"]["total_fixtures"] == 1


def test_scan_rejects_bad_thresholds():
    assert client.post("/scan", json={"raw_text": TXT, "min_threshold": 0.9, "threshold": 0.5}).status_code == 422


def test_ui_escapes_html():
    r = client.post("/ui-scan", data={"raw_text": "X: Y\n15:00 <script>alert(1)</script> - Foo"})
    assert "<script>alert(1)" not in r.text


def test_health():
    assert client.get("/health").json()["status"] == "ok"
