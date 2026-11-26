import numpy as np
import pytest

from scanner.model import (Rating, Ratings, dixon_coles_matrix,
                           evaluate_triple_markets, expected_goals)


def test_matrix_normalised_and_nonnegative():
    m = dixon_coles_matrix(1.485, 1.2285)
    assert m.sum() == pytest.approx(1.0) and (m >= 0).all()


def test_pmf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    from scanner.model import _poisson_pmf
    assert np.allclose(_poisson_pmf(2.3, 10), stats.poisson.pmf(np.arange(10), 2.3), atol=1e-12)


def test_unrated_lambdas_match_original_constants():
    lh, la = expected_goals(None, None)
    assert lh == pytest.approx(1.485) and la == pytest.approx(1.2285)


def test_stronger_attack_raises_lambda_and_probs_move():
    lh, la = expected_goals(Rating(attack=1.4), Rating(defence=0.8))
    base_h, _ = expected_goals(None, None)
    assert lh > base_h
    assert evaluate_triple_markets(lh, la)["under_2_5_prob"] < evaluate_triple_markets(1.485, 1.2285)["under_2_5_prob"]


def test_ratings_lookup_case_insensitive(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"_comment":"x","Arsenal":{"attack":1.3,"defence":0.8}}')
    r = Ratings.from_json(p)
    assert len(r) == 1 and r.get("  arsenal ").attack == 1.3


def test_ratings_out_of_range_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"X":{"attack":9}}')
    with pytest.raises(ValueError):
        Ratings.from_json(p)


def test_missing_ratings_file_is_empty(tmp_path):
    assert len(Ratings.from_json(tmp_path / "nope.json")) == 0


def test_odds_roundtrip_recovers_lambdas():
    from scanner.model import lambdas_from_1x2
    for lh, la in [(1.4, 1.1), (2.2, 0.6), (0.9, 1.6)]:
        m = dixon_coles_matrix(lh, la)
        x, y = np.indices(m.shape)
        odds = tuple(round(1 / (p * 1.06), 2) for p in (m[x > y].sum(), m[x == y].sum(), m[x < y].sum()))
        got = lambdas_from_1x2(odds)
        assert abs(got[0] - lh) < 0.05 and abs(got[1] - la) < 0.05


def test_devig_sums_to_one():
    from scanner.model import devig
    assert sum(devig((2.0, 3.4, 3.8))) == pytest.approx(1.0)


def test_all_three_is_1x_and_under25_and_below_min():
    m = evaluate_triple_markets(1.3, 1.0)
    assert 0 < m["all_three_prob"] <= m["min_prob"] + 1e-12
    mat = dixon_coles_matrix(1.3, 1.0)
    x, y = np.indices(mat.shape)
    assert m["all_three_prob"] == pytest.approx(mat[(x >= y) & (x + y <= 2)].sum())


def test_accent_and_punctuation_insensitive_lookup():
    r = Ratings({"St. Truiden": Rating(1.1, 0.9)})
    assert r.get("st truiden") is not None and r.get("Sint-Truiden") is None
