import pytest
import numpy as np

from src.eval import metrics


def test_ndcg_k_is_zero():
    relevant = [1, 2, 3]
    relevancy_scores = [2, 2, 1]
    predicted = [1, 3, 2]
    with pytest.raises(Exception):
        metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=0)


def test_ndcg_k_larger_than_predicted():
    relevant = [1, 2, 3]
    relevancy_scores = [2, 2, 1]
    predicted = [1, 3, 2]
    with pytest.raises(Exception):
        metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=len(predicted) + 1)


def test_ndcg_k_none():
    relevant = [1, 2, 3]
    relevancy_scores = [0.2, 0.3, 0.5]
    predicted = [3, 2, 1, 4, 10, 11]
    ndcg_3 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=6)
    ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted)

    assert ndcg_3 == ndcg_score


def test_ndcg_k_no_hits():
    relevant = [1, 2, 3]
    relevancy_scores = [0.2, 0.3, 0.5]
    predicted = [4, 5, 6, 7, 8, 9, 10]
    for i in range(len(predicted)):
        ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=i + 1)
        assert ndcg_score == 0.0


def test_ndcg_k_perfect():
    relevant = [1, 2, 3]
    relevancy_scores = [0.2, 0.3, 0.5]
    predicted = [3, 2, 1]
    for i in range(len(predicted)):
        ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=i + 1)
        assert ndcg_score == 1.0


def test_ndcg_k_reals():
    relevant = [4, 2, 1]
    relevancy_scores = [0.7, 0.2, 0.1]
    predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ndcg_10 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=10)
    ndcg_3 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=3)
    ndcg_1 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=1)

    assert np.isclose(ndcg_10, 0.602227, atol=1e-4)
    assert np.isclose(ndcg_3, 0.258148, atol=1e-4)
    assert np.isclose(ndcg_1, 0.14285, atol=1e-4)


def test_ndcg_k_reals_one_hit():
    relevant = [4, 2, 1, 0]
    relevancy_scores = [0.7, 0.1, 0.05, 0.15]
    predicted = [7, 3, 10, 11, 12, 13, 14, 4]
    ndcg_8 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=8)

    assert np.isclose(ndcg_8, 0.254943, atol=1e-4)

    for i in range(1, len(predicted)):
        ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=i)
        assert ndcg_score == 0.0


# https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
def test_ndcg_k_integers_from_wikipedia():
    relevant = [1, 2, 3, 5, 6]
    relevancy_scores = [3, 2, 3, 1, 2]
    predicted = [1, 2, 3, 4, 5, 6]
    k = 6
    ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=k)

    assert np.isclose(ndcg_score, 0.9607, atol=1e-4)


def test_ndcg_k_integers():
    relevant = [1, 2, 3]
    relevancy_scores = [2, 2, 1]
    predicted = [1, 3, 2]
    ndcg_3 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=3)
    ndcg_2 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=2)
    ndcg_1 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=1)
    assert np.isclose(ndcg_3, 0.9651, atol=1e-4)
    assert np.isclose(ndcg_2, 0.8065, atol=1e-4)
    assert np.isclose(ndcg_1, 1, atol=1e-4)


def test_ndcg_k_one_hit():
    relevant = [2, 3, 4]
    relevancy_scores = [2, 2, 3]
    predicted = [0, 1, 10, 2]
    ndcg_4 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=4)
    ndcg_1 = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=1)
    assert np.isclose(ndcg_4, 0.163698, atol=1e-4)
    assert ndcg_1 == 0.0


def test_ndcg_k_zero():
    relevant = [1]
    relevancy_scores = [10]
    predicted = [5, 6]
    k = 2
    ndcg_score = metrics.ndcg_k(relevant=relevant, relevancy_scores=relevancy_scores, predicted=predicted, k=k)

    assert ndcg_score == 0.0
