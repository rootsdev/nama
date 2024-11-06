import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpire import WorkerPool
from scipy.integrate import cumulative_trapezoid


def precision_k(actuals: list, candidates: list, k: int) -> float:
    """
    Return the precision at k given a list of actuals and an ordered list of candidates
    """
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actuals).intersection(candidates)) / min(k, len(candidates))


def recall_k(actuals: list, candidates: list, k: int) -> float:
    """
    Return the recall at k given a list of actuals and an ordered list of candidates
    """
    if len(candidates) > k:
        candidates = candidates[:k]
    return len(set(actuals).intersection(candidates)) / len(actuals)


def precision_recall_at_k(
    actuals_list: list[list], candidates_list: list[list], max_k: int
) -> (list[float], list[float]):
    """
    Return a list of average precisions and recalls for 1..max_k for the given actuals and candidates
    """
    precisions = []
    recalls = []
    for i in range(max_k):
        precisions.append(np.mean([precision_k(a, c, i + 1) for a, c in zip(actuals_list, candidates_list)]))
        recalls.append(np.mean([recall_k(a, c, i + 1) for a, c in zip(actuals_list, candidates_list)]))
    return precisions, recalls


def precision_recall_curve_at_k(actuals_list: list[list], candidates_list: list[list], max_k: int):
    """
    Plot a precision-recall curve for 1..max_k for the given actuals and candidates
    """
    show_precision_recall_curve(*precision_recall_at_k(actuals_list, candidates_list, max_k))


def show_precision_recall_curve(precisions: list[float], recalls: list[float]):
    """
    Plot a precision-recall curve for the given precisions and recalls
    """
    plt.plot(recalls, precisions, "ko--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def _get_matches(candidates: np.ndarray, threshold: float, distances: bool = False):
    return set(
        candidates[candidates[:, 1] <= threshold][:, 0]
        if distances
        else candidates[candidates[:, 1] >= threshold][:, 0]
    )


def precision_at_threshold(
    record_name_frequencies: list[tuple[str, int]],
    candidates: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the precision at a threshold for the given weighted-actuals and candidates
    :param record_name_frequencies: list of [name, frequency] - frequency is ignored
    :param candidates: array of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    matches = _get_matches(candidates, threshold, distances)
    num_matches = len(matches)
    if num_matches == 0:
        return 1.0
    return len({name for name, _ in record_name_frequencies}.intersection(matches)) / num_matches


def weighted_recall_at_threshold(
    record_name_frequencies: list[tuple[str, int]],
    candidates: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the weighted-recall at a threshold for the given weighted-actuals and candidates,
    where weighted-recall is the sum of the weights of each matched name
    :param record_name_frequencies: list of [name, frequency]
    :param candidates: array of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    matches = _get_matches(candidates, threshold, distances)
    if len(record_name_frequencies) == 0:
        return 1.0
    total_freq = sum(freq for _, freq in record_name_frequencies)
    weighted_recall = sum(freq / total_freq for name, freq in record_name_frequencies if name in matches)
    if weighted_recall > 1.0001:
        raise Exception("Impossible recall")
    return weighted_recall


def avg_precision_at_threshold(
    record_name_frequencies_list: list[list[tuple[str, int]]],
    candidates_list: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the average precision at a threshold for a list of weighted-actuals and a list of candidates.
    Like precision_at_threshold, but operates on lists of lsits, where each entry in the two lists has the
    record_name_frequencies and candidates for a particular name
    :param record_name_frequencies_list: list of lists of [name, weight, ?]
    :param candidates_list: list of list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    :return:
    """
    avg_precisions = []
    for a, c in zip(record_name_frequencies_list, candidates_list):
        avg_precisions.append(precision_at_threshold(a, c, threshold, distances))
    return np.mean(avg_precisions)


def avg_weighted_recall_at_threshold(
    record_name_frequencies_list: list[list[tuple[str, int]]],
    candidates_list: np.ndarray,
    threshold: float,
    distances: bool = False,
) -> float:
    """
    Return the average weighted-recall at a threshold of a list of weighted-actuals and a list of candidates
    :param record_name_frequencies_list: list of lists of [name, frequency]
    :param candidates_list: list of list of [name, score]
    :param threshold: threshold
    :param distances: if True, score must be <= threshold; if False, score must be >= threshold; defaults to False
    """
    avg_recalls = []
    for a, c in zip(record_name_frequencies_list, candidates_list):
        avg_recalls.append(weighted_recall_at_threshold(a, c, threshold, distances))
    return np.mean(avg_recalls)


def precision_weighted_recall_curve_at_threshold(
    record_name_frequencies_list: list[list[tuple[str, int]]],
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
            record_name_frequencies_list, candidates_list, min_threshold, max_threshold, step, distances
        )
    )


def precision_weighted_recall_at_threshold(
    record_name_frequencies_list: list[list[tuple[str, int]]],
    candidates_list: np.ndarray,
    min_threshold: float = 0.5,
    max_threshold: float = 1.0,
    step=0.01,
    distances=False,
    n_jobs=1,
    progress_bar=False,
) -> (list[float], list[float]):
    """
    Return lists of average precisions and average weighted-recalls for threshold: min_threshold..max_threshold, with step
    """

    def get_precision_recall(shared, threshold):
        record_name_frequencies_list, candidates_list, distances = shared
        precision = np.mean([
            precision_at_threshold(a, c, threshold, distances)
            for a, c in zip(record_name_frequencies_list, candidates_list)
        ])
        recall = np.mean([
            weighted_recall_at_threshold(a, c, threshold, distances)
            for a, c in zip(record_name_frequencies_list, candidates_list)
        ])
        return precision, recall, threshold

    thresholds = list(np.arange(min_threshold, max_threshold, step))
    if n_jobs == 1:
        precisions_recalls = [
            get_precision_recall((record_name_frequencies_list, candidates_list, distances), threshold)
            for threshold in thresholds
        ]
    else:
        with WorkerPool(
            shared_objects=(record_name_frequencies_list, candidates_list, distances), n_jobs=n_jobs
        ) as pool:
            precisions_recalls = pool.map(get_precision_recall, thresholds, progress_bar=progress_bar)

    precisions_recalls.sort(key=lambda tup: tup[2])
    precisions = [precision for precision, _, _ in precisions_recalls]
    recalls = [recall for _, recall, _ in precisions_recalls]

    return precisions, recalls


def get_auc_from_precisions_recalls(
    precisions: list[float],
    recalls: list[float],
    distances: bool = False,
) -> float:
    """
    Return the area under the curve of precision and weighted-recall
    """
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
    for ix, (p, r) in enumerate(zip(precisions, recalls)):
        if ix > 0 and (r < prev_r or math.isclose(r, prev_r, abs_tol=1e-4)):
            continue
        precs.append(p)
        recs.append(r)
        prev_r = r
    if len(precs) < 2:
        return 0
    # switch to cumulative_trapezoid because simpson over the following unexpectedly returns 1.746
    # precs = [1.000000000, 1.000000000, 0.000212482]
    # recs = [0.000000000, 0.104290700, 0.105383575]
    # auc = scipy.integrate.simpson(precs, recs)
    auc = cumulative_trapezoid(precs, recs, initial=0)[-1]
    if auc < 0.0 or auc > 1.0001:
        precisions = ",".join([f"{i:.9f}" for i in precs])
        recalls = ",".join([f"{i:.9f}" for i in recs])
        raise Exception(f"Invalid AUC precs={precisions} recs={recalls}")
    return auc


def get_auc(
    record_name_frequencies_list: list[list[tuple[str, int]]],
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
        record_name_frequencies_list, candidates_list, min_threshold, max_threshold, step, distances
    )
    return get_auc_from_precisions_recalls(precisions, recalls, distances)


def ndcg_k(relevant: list, relevancy_scores: list, predicted: list, k: Optional[int] = None) -> float:
    """
    Computes Normalized Discount Cumulative Gain (nDCG) at cut-off k.
    :param relevant: The list of relevant items.
    :param relevancy_scores: The relevancy scores for each of the relevant items. Must be in the same order as the
    relevant item list.
    :param predicted: The items that are predicted to be relevant by the model sorted by descending order of relevancy.
    :param k: Up to what position to calculate nDCG for the predicted items. If k is None it will be calculated for
    the full predicted list.
    :return:
    """
    if k is None:
        k = len(predicted)
    elif k > len(predicted):
        raise ValueError("k cannot be larger than number of elements in predicted")
    elif k == 0:
        raise ValueError("k must be greater than 0")
    # assert len(relevant) == len(relevancy_scores)
    predicted_item_relevancy_scores = _get_predicted_item_relevancy_scores(relevant, relevancy_scores, predicted)
    dcg = _dcg_k(predicted_item_relevancy_scores, k)
    idcg = _dcg_k(np.sort(relevancy_scores)[::-1], k)
    ndcg = float(dcg / idcg)
    return ndcg


def _dcg_k(relevancy_scores, k):
    discounts = 1.0 / np.log2(np.arange(1, min(k, len(relevancy_scores)) + 1) + 1)
    return np.sum(relevancy_scores[:k] * discounts)


def _get_predicted_item_relevancy_scores(relevant: list, relevancy_scores: list, predicted: list) -> np.ndarray:
    """
    Creates a relevancy array for the predicted items using the ground truth relevant items and scores.
    Example:
        relevant = [1, 2, 3]
        relevancy_scores = [3, 1, 1]
        predicted = [1, 4, 5, 10, 2, 99]

        _get_predicted_item_relevancy_scores(relevant, relevancy_scores, predicted) -> [3, 0, 0, 0, 1, 0]
    """
    relevancy_scores_dict = dict(zip(relevant, relevancy_scores))
    predicted_item_relevancy_scores = []
    for item in predicted:
        if item in relevant:
            predicted_item_relevancy_scores.append(relevancy_scores_dict[item])
        else:
            predicted_item_relevancy_scores.append(0)
    predicted_item_relevancy_scores = np.array(predicted_item_relevancy_scores)
    # assert len(predicted_item_relevancy_scores) == len(predicted)

    return predicted_item_relevancy_scores
