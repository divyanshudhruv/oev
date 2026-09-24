import torch

from oev.benchmark_ext import (
    aurc,
    confident_error_rate,
    coverage_at_error_budget,
)


def test_confident_error_rate_perfect():
    # 2 answers at p >= 0.9, both correct -> zero confident errors
    confs = [0.99, 0.6, 0.95, 0.3]
    corrs = [1, 0, 1, 1]
    rate, n = confident_error_rate(confs, corrs)
    assert rate == 0.0
    assert n == 2


def test_confident_error_rate_with_errors():
    confs = [0.95, 0.92, 0.5]
    corrs = [0, 1, 0]
    rate, n = confident_error_rate(confs, corrs)
    assert n == 2
    assert abs(rate - 1 / 3) < 1e-9  # 1 confident error among 3 answers


def test_confident_error_rate_threshold():
    confs = [0.89, 0.91]
    corrs = [0, 0]
    rate, n = confident_error_rate(confs, corrs, threshold=0.9)
    assert n == 1
    assert abs(rate - 0.5) < 1e-9


def test_coverage_perfect_predictor():
    # a 0.6-confident wrong answer must be excluded; the rest automate
    confs = [0.99, 0.6, 0.95, 0.3]
    corrs = [1, 0, 1, 1]
    cov = coverage_at_error_budget(confs, corrs, 0.05)
    assert cov == 0.5


def test_coverage_all_correct():
    confs = [0.8, 0.7, 0.6, 0.55]
    corrs = [1, 1, 1, 1]
    cov = coverage_at_error_budget(confs, corrs, 0.05)
    assert cov == 1.0


def test_coverage_never_ok_when_first_answer_wrong():
    # the highest-confidence answer is wrong, so any threshold including
    # it breaks the budget; thresholds below it admit the same error
    confs = [0.99, 0.9, 0.8]
    corrs = [0, 1, 1]
    cov = coverage_at_error_budget(confs, corrs, 0.05)
    assert cov == 0.0


def test_aurc_bounds():
    # perfect predictions: risk stays 0 -> AURC 0
    assert aurc([0.9, 0.8, 0.7], [1, 1, 1]) == 0.0
    # all wrong: risk is 1 at every coverage -> AURC 1
    assert aurc([0.9, 0.8, 0.7], [0, 0, 0]) == 1.0
    # AURC of the mixed case sits between the extremes
    mixed = aurc([0.9, 0.6], [1, 0])
    assert 0.0 < mixed < 1.0
