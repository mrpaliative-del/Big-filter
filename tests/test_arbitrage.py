import pytest

from arbitrage import find_arbitrage


def test_three_way_arbitrage_locks_equal_payout():
    q = {"home": [("A", 2.40), ("B", 2.10)], "draw": [("A", 3.30), ("B", 3.60)], "away": [("A", 3.10), ("B", 4.00)]}
    r = find_arbitrage(q, 100)
    assert r["arbitrage"] and r["overround"] < 1 and r["profit_pct"] > 0
    assert r["best"]["away"] == {"bookmaker": "B", "price": 4.0}
    assert sum(r["stakes"].values()) == pytest.approx(100)
    assert max(r["payout"].values()) - min(r["payout"].values()) < 1e-9          # same return whatever happens
    assert r["payout"]["home"] > 100


def test_no_arbitrage_when_margin_positive():
    q = {"home": [("A", 2.0)], "draw": [("A", 3.4)], "away": [("A", 3.8)]}
    r = find_arbitrage(q)
    assert not r["arbitrage"] and r["overround"] > 1 and r["profit_pct"] < 0


def test_two_way_market():
    r = find_arbitrage({"over": [("A", 2.10)], "under": [("B", 2.05)]})
    assert r["arbitrage"] and r["profit_pct"] == pytest.approx((1 / (1 / 2.10 + 1 / 2.05) - 1) * 100)
