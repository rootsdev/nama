import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy


def precision_k(actuals: list, candidates: list, k: int) -> float:
    """
    Return the precision at k given a list of actuals and an ordered list of candidates
    """
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actuals).intersection(candidates)) / min(k, len(candidates))


def avg_precision_k(actuals: list, candidates: list, max_k: int) -> (float, List[float]):
    """
    Return the average precision and the list of precisions from 1..max_k given a list of actuals and an ordered list of candidates
    """
    precisions = []
    for i in range(max_k):
        precisions.append(precision_k(actuals, candidates, i + 1))

    return np.mean(precisions), precisions


def mean_avg_precision_k(actuals_list: List[list], candidates_list: List[list], max_k: int) -> float:
    """
    Return the mean average precision from 1..max_k given a list of actuals lists and a list of ordered candidates lists
    :param actuals_list: list of lists of names
    :param candidates_list: list of ordered lists of names
    :param max_k: maximum value of k
    """
    avg_precisions = []
    for a, c in zip(actuals_list, candidates_list):
        avg_precisions.append(avg_precision_k(a, c, max_k)[0])

    return np.mean(avg_precisions)


def recall_k(actuals: list, candidates: list, k: int) -> float:
    """
    Return the recall at k given a list of actuals and an ordered list of candidates
    """
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actuals).intersection(candidates)) / len(actuals)


def precision_recall_at_k(
    actuals_list: List[list], candidates_list: List[list], max_k: int
) -> (List[float], List[float]):
    """
    Return a list of average precisions and recalls for 1..max_k for the given actuals and candidates
    """
    precisions = []
    recalls = []
    for i in range(max_k):
        precisions.append(np.mean([precision_k(a, c, i + 1) for a, c in zip(actuals_list, candidates_list)]))
        recalls.append(np.mean([recall_k(a, c, i + 1) for a, c in zip(actuals_list, candidates_list)]))
    return precisions, recalls


def precision_recall_curve_at_k(actuals_list: List[list], candidates_list: List[list], max_k: int):
    """
    Plot a precision-recall curve for 1..max_k for the given actuals and candidates
    """
    show_precision_recall_curve(*precision_recall_at_k(actuals_list, candidates_list, max_k))


def show_precision_recall_curve(precisions: List[float], recalls: List[float]):
    """
    Plot a precision-recall curve for the given precisions and recalls
    """
    plt.plot(recalls, precisions, "ko--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def precision_at_threshold(
    weighted_actual_names: List[Tuple[str, float, int]],
    candidates: List[Tuple[str, float]],
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the precision at a threshold for the given weighted-actuals and candidates
    :param weighted_actual_names: list of [name, weight, ?] - weight and ? are ignored
    :param candidates: list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    matches = (
        candidates[candidates[:, 1] <= threshold][:, 0]
        if distances
        else candidates[candidates[:, 1] >= threshold][:, 0]
    )
    num_matches = len(matches)
    if num_matches == 0:
        return 1.0
    return len(set(name for name, weight, _ in weighted_actual_names).intersection(matches)) / num_matches


def weighted_recall_at_threshold(
    weighted_actual_names: List[Tuple[str, float, int]],
    candidates: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the weighted-recall at a threshold for the given weighted-actuals and candidates,
    where weighted-recall is the sum of the weights of each matched name
    :param weighted_actual_names: list of [name, weight, ?] - weight and ? are ignored
    :param candidates: list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    matches = (
        candidates[candidates[:, 1] <= threshold][:, 0]
        if distances
        else candidates[candidates[:, 1] >= threshold][:, 0]
    )
    return sum(weight for name, weight, _ in weighted_actual_names if name in matches)


def avg_precision_at_threshold(
    weighted_actual_names_list: List[List[Tuple[str, float, int]]],
    candidates_list: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the average precision at a threshold for a list of weighted-actuals and a list of candidates.
    Like precision_at_threshold, but operates on lists of lsits, where each entry in the two lists has the
    weighted_actual_names and candidates for a particular name
    :param weighted_actual_names_list: list of lists of [name, weight, ?]
    :param candidates_list: list of list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    :return:
    """
    avg_precisions = []
    for a, c in zip(weighted_actual_names_list, candidates_list):
        avg_precisions.append(precision_at_threshold(a, c, threshold, distances))
    return np.mean(avg_precisions)


def avg_weighted_recall_at_threshold(
    weighted_actual_names_list: List[List[Tuple[str, float, int]]],
    candidates_list: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the average weighted-recall at a threshold of a list of weighted-actuals and a list of candidates
    :param weighted_actual_names_list: list of lists of [name, weight, ?]
    :param candidates_list: list of list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    avg_recalls = []
    for a, c in zip(weighted_actual_names_list, candidates_list):
        avg_recalls.append(weighted_recall_at_threshold(a, c, threshold, distances))
    return np.mean(avg_recalls)


def precision_weighted_recall_curve_at_threshold(
    weighted_actual_names_list: List[List[Tuple[str, float, int]]],
    candidates_list: np.ndarray,
    min_threshold: float = 0.5,
    max_threshold: float = 1.0,
    step=0.01,
    distances=False,
):
    """
    Plot precision-weighted-recall curve for threshold: min_threshold..max_threshold, with step
    :return:
    """
    show_precision_recall_curve(
        *precision_weighted_recall_at_threshold(
            weighted_actual_names_list, candidates_list, min_threshold, max_threshold, step, distances
        )
    )


def precision_weighted_recall_at_threshold(
    weighted_actual_names_list: List[List[Tuple[str, float, int]]],
    candidates_list: np.ndarray,
    min_threshold: float = 0.5,
    max_threshold: float = 1.0,
    step=0.01,
    distances=False,
) -> (List[float], List[float]):
    """
    Return lists of average precisions and average weighted-recalls for threshold: min_threshold..max_threshold, with step
    """
    precisions = []
    recalls = []
    for i in np.arange(min_threshold, max_threshold, step):
        precisions.append(
            np.mean(
                [
                    precision_at_threshold(a, c, i, distances)
                    for a, c in zip(weighted_actual_names_list, candidates_list)
                ]
            )
        )
        recalls.append(
            np.mean(
                [
                    weighted_recall_at_threshold(a, c, i, distances)
                    for a, c in zip(weighted_actual_names_list, candidates_list)
                ]
            )
        )
    return precisions, recalls


def get_auc(
    weighted_actual_names_list: List[List[Tuple[str, float, int]]],
    candidates_list: np.ndarray,
    min_threshold: float = 0.5,
    max_threshold: float = 1.0,
    step: float = 0.01,
    distances: bool = False,
) -> float:
    """
    Return the area under the curve of precision and weighted-recall
    """
    precisions, recalls = precision_weighted_recall_at_threshold(
        weighted_actual_names_list, candidates_list, min_threshold, max_threshold, step, distances
    )
    # start with low recall, high precision
    if not distances:
        precisions.reverse()
        recalls.reverse()
    # take recall all the way to 0 with earliest precision so the curve starts at 0
    precisions.insert(0, precisions[0])
    recalls.insert(0, 0.0)
    # remove points that have nearly the same recall so simpson doesn't blow up
    precs = []
    recs = []
    prev_r = float("nan")
    for p, r in zip(precisions, recalls):
        if math.isclose(r, prev_r):
            continue
        precs.append(p)
        recs.append(r)
        prev_r = r
    return scipy.integrate.simpson(precs, recs)
