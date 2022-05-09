import jellyfish
from mpire import WorkerPool
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from src.models.utils import remove_padding


def calc_lev_similarity(name, cand_name):
    dist = jellyfish.levenshtein_distance(name, cand_name)
    # dist = levenshtein(name, cand_name)
    return 1 - (dist / max(len(name), len(cand_name)))


def _calc_lev_similarity_to(name):
    name = remove_padding(name)

    def calc_similarity(row):
        cand_name = remove_padding(row[0])
        return calc_lev_similarity(name, cand_name)

    return calc_similarity


def _get_similars(shared, batch, _=None):
    tfidf_vectorizer, candidate_names, candidate_names_X, k = shared

    # get the tfidf vectors for the input names
    batch_X = tfidf_vectorizer.transform(batch)

    def get_similars_for_name(name, name_X):
        scores = np.zeros(candidate_names_X.shape[0])
        # get top k candidate_names
        tfidf_scores = np.squeeze(safe_sparse_dot(candidate_names_X, name_X.T).toarray(), axis=1)
        ixs = np.argpartition(tfidf_scores, -k)[-k:]

        # get levenshtein scores for top k candidate names
        scores[ixs] = np.apply_along_axis(_calc_lev_similarity_to(name), 1, candidate_names[ixs, None])

        # return sorted scores
        partitioned_idx = np.argpartition(scores, -k)[-k:]
        sorted_partitioned_idx = np.argsort(scores[partitioned_idx])[::-1]
        sorted_scores_idx = partitioned_idx[sorted_partitioned_idx]

        sorted_names = candidate_names[sorted_scores_idx]
        sorted_scores = scores[sorted_scores_idx]

        return np.stack((np.array(sorted_names).astype(object), np.array(sorted_scores)), axis=1)

    result = []
    for name, name_X in zip(batch, batch_X):
        result.append(get_similars_for_name(name, name_X))
    return np.stack(result, axis=0)


def get_best_lev_matches(tfidf_vectorizer, input_names, candidate_names, k, batch_size=512, n_jobs=1, progress_bar=True):
    """
    Get the best lev matches where candidate names is a list of all candidate names.
    """
    # get the tfidf vectors for the candidate names
    candidate_names_X = tfidf_vectorizer.transform(candidate_names)
    # generate batches
    batches = []
    for ix in range(0, len(input_names), batch_size):
        # chunks needs to be a list of tuples; otherwise mpire passes each row in the chunk as a separate parameter
        batches.append((input_names[ix:ix + batch_size], ix))
    if n_jobs == 1:
        results = []
        if progress_bar:
            batches = tqdm(batches)
        for batch, ix in batches:
            results.append(_get_similars((tfidf_vectorizer, candidate_names, candidate_names_X, k), batch, ix))
        candidate_names_scores = np.vstack(results)
    else:
        with WorkerPool(shared_objects=(tfidf_vectorizer, candidate_names, candidate_names_X, k), n_jobs=n_jobs) as pool:
            candidate_names_scores = pool.map(_get_similars, batches, progress_bar=progress_bar)
    return candidate_names_scores


def _get_lev_scores(input_names, candidate_names):
    result = []
    for name, candidates in zip(input_names, candidate_names):
        result.append(np.apply_along_axis(_calc_lev_similarity_to(name), 1, candidates[:, None]))
    return np.stack(result, axis=0)


def get_lev_scores(input_names, candidate_names, batch_size=512, n_jobs=1, progress_bar=True):
    """
    Get the lev scores
    :param input_names: a 1d array of names
    :param candidate_names: a 2d array of matching names for each input name
    :returns candidate_scores: a 2d array of lev scores for each input-name candidate-name pair
    """
    # generate batches
    batches = []
    for ix in range(0, len(input_names), batch_size):
        batches.append((input_names[ix:ix + batch_size], candidate_names[ix:ix + batch_size]))
    if n_jobs == 1:
        results = []
        if progress_bar:
            batches = tqdm(batches)
        for names, candidates in batches:
            results.append(_get_lev_scores(names, candidates))
        candidate_scores = np.vstack(results)
    else:
        with WorkerPool(n_jobs=n_jobs) as pool:
            candidate_scores = pool.map(_get_lev_scores, batches, progress_bar=progress_bar)
    return candidate_scores
