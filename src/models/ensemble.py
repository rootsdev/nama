import math

import jellyfish
import numpy as np
# from rapidfuzz.string_metric import levenshtein
from tqdm import tqdm

from src.models.levenshtein import get_lev_scores, get_best_lev_matches
from src.models.swivel import get_best_swivel_matches
from src.models.utils import remove_padding


def featurize(swivel_score, lev_score, input_name_freq, candidate_name_freq, is_oov):
    log_max_freq = math.log10(max(input_name_freq, candidate_name_freq)+1)
    log_min_freq = math.log10(min(input_name_freq, candidate_name_freq)+1)
    return [
        swivel_score if not is_oov else 0.0,
        swivel_score * log_max_freq if not is_oov else 0.0,
        swivel_score * log_min_freq if not is_oov else 0.0,
        lev_score if not is_oov else 0.0,
        lev_score * log_max_freq if not is_oov else 0.0,
        lev_score * log_min_freq if not is_oov else 0.0,
        1.0 if not is_oov else 0.0,

        lev_score if is_oov else 0.0,
        lev_score * log_max_freq if is_oov else 0.0,
        lev_score * log_min_freq if is_oov else 0.0,
        1.0 if is_oov else 0.0,
    ]


def _calc_lev_similarity_to(name):
    name = remove_padding(name)

    def calc_similarity(row):
        cand_name = remove_padding(row[0])
        dist = jellyfish.levenshtein_distance(name, cand_name)
        # dist = levenshtein(name, cand_name)
        return 1 - (dist / max(len(name), len(cand_name)))

    return calc_similarity


def _get_lev_similars_for_name(name, candidate_names):
    scores = np.apply_along_axis(_calc_lev_similarity_to(name), 1, candidate_names[:, None])

    sorted_scores_idx = np.argsort(scores)[::-1]
    candidate_names = candidate_names[sorted_scores_idx]
    candidate_scores = scores[sorted_scores_idx]

    return list(zip(candidate_names, candidate_scores))


def _get_ensemble_names_scores(ensemble_model, name_freq, input_names, k,
                               swivel_names_scores_in_vocab, lev_scores_in_vocab,
                               lev_names_scores_out_vocab, verbose=True):
    result = []
    # get best ensemble names+scores for each input name
    rng = range(input_names.shape[0])
    if verbose:
        rng = tqdm(rng)
    for i in rng:
        input_name = input_names[i]
        input_name_freq = name_freq.get(input_name, 0)
        features = []
        candidate_names = []
        if swivel_names_scores_in_vocab is not None:
            # add features for swivel+lev scores
            for swivel_name_score, lev_score in zip(swivel_names_scores_in_vocab[i], lev_scores_in_vocab[i]):
                candidate_name = swivel_name_score[0]
                candidate_name_freq = name_freq.get(candidate_name, 0)
                swivel_score = swivel_name_score[1]
                candidate_names.append(candidate_name)
                features.append(featurize(
                    swivel_score,
                    lev_score,
                    input_name_freq,
                    candidate_name_freq,
                    is_oov=False,
                ))
        if lev_names_scores_out_vocab is not None:
            # add features for just lev scores
            for lev_name_score in lev_names_scores_out_vocab[i]:
                candidate_name = lev_name_score[0]
                candidate_name_freq = name_freq.get(candidate_name, 0)
                lev_score = lev_name_score[1]
                candidate_names.append(candidate_name)
                features.append(featurize(
                    0.0,
                    lev_score,
                    input_name_freq,
                    candidate_name_freq,
                    is_oov=True,
                ))
        # predict
        predictions = ensemble_model.predict_proba(features)[:, 1]
        candidate_names = np.array(candidate_names, dtype="O")
        if k < len(predictions):
            # get indices of the top k predictions
            ixs = np.argpartition(predictions, -k)[-k:]
            # return np.stack(names, predictions)
            top_names = candidate_names[ixs]
            top_scores = predictions[ixs]
        else:
            top_names = candidate_names
            top_scores = predictions
        result.append(np.stack((top_names, top_scores), axis=1))
    return np.stack(result, axis=0)


def get_best_ensemble_matches(model, vocab, name_freq, input_names, candidate_names, tfidf_vectorizer, ensemble_model,
                              k, batch_size, add_context=True, n_jobs=1, verbose=True):
    # get in-vocab vs out-of-vocab input names and candidate names
    vocab_names = set(vocab.keys())
    input_names = np.asarray(input_names)
    # input_in_vocab_ixs = np.isin(input_names, vocab_names)
    input_in_vocab_ixs = np.array([name in vocab_names for name in input_names])
    input_out_vocab_ixs = np.invert(input_in_vocab_ixs)
    candidate_names = np.asarray(candidate_names)
    # candidate_in_vocab_ixs = np.isin(candidate_names, vocab_names)
    candidate_in_vocab_ixs = np.array([name in vocab_names for name in candidate_names])
    candidate_out_vocab_ixs = np.invert(candidate_in_vocab_ixs)

    # initialize empty ensemble_names_scores result
    if k > candidate_names.shape[0]:
        k = candidate_names.shape[0]
    ensemble_names_scores_names = np.empty((input_names.shape[0], k), dtype="O")
    ensemble_names_scores_scores = np.empty((input_names.shape[0], k), dtype=float)
    ensemble_names_scores = np.dstack((ensemble_names_scores_names, ensemble_names_scores_scores))

    if k == 0:
        return ensemble_names_scores

    if np.sum(input_in_vocab_ixs) > 0:
        # get swivel scores for input name & candidate name pairs that are both in-vocab
        n_candidate_in_vocab = np.sum(candidate_in_vocab_ixs)
        if n_candidate_in_vocab > 0:
            swivel_names_scores_in_vocab = get_best_swivel_matches(model=model,
                                                                   vocab=vocab,
                                                                   input_names=input_names[input_in_vocab_ixs],
                                                                   candidate_names=candidate_names[candidate_in_vocab_ixs],
                                                                   k=min(n_candidate_in_vocab, k),
                                                                   batch_size=batch_size,
                                                                   add_context=add_context,
                                                                   n_jobs=n_jobs,
                                                                   progress_bar=verbose)
            # get lev scores for in-vocab name pairs
            lev_scores_in_vocab = get_lev_scores(input_names[input_in_vocab_ixs],
                                                 swivel_names_scores_in_vocab[:, :, 0],
                                                 batch_size=batch_size,
                                                 n_jobs=n_jobs,
                                                 progress_bar=verbose)
        else:
            swivel_names_scores_in_vocab = None
            lev_scores_in_vocab = None

        # get lev scores for input name & candidate name pairs where the input names are in-vocab, candidate names are out
        n_candidate_out_vocab = np.sum(candidate_out_vocab_ixs)
        if n_candidate_out_vocab > 0:
            lev_names_scores_candidate_out_vocab = get_best_lev_matches(tfidf_vectorizer,
                                                                        input_names[input_in_vocab_ixs],
                                                                        candidate_names[candidate_out_vocab_ixs],
                                                                        k=min(n_candidate_out_vocab, k),
                                                                        batch_size=batch_size,
                                                                        n_jobs=n_jobs,
                                                                        progress_bar=verbose)
        else:
            lev_names_scores_candidate_out_vocab = None

        # get in-vocab ensemble names and scores
        ensemble_names_scores[input_in_vocab_ixs] = _get_ensemble_names_scores(ensemble_model,
                                                                               name_freq,
                                                                               input_names[input_in_vocab_ixs],
                                                                               k,
                                                                               swivel_names_scores_in_vocab,
                                                                               lev_scores_in_vocab,
                                                                               lev_names_scores_candidate_out_vocab,
                                                                               verbose=verbose)

    if np.sum(input_out_vocab_ixs) > 0:
        # get lev matches for input name & candidate name pairs where the input names are out-of-vocab
        lev_names_scores_input_out_vocab = get_best_lev_matches(tfidf_vectorizer,
                                                                input_names[input_out_vocab_ixs],
                                                                candidate_names,
                                                                k=k,
                                                                batch_size=batch_size,
                                                                n_jobs=n_jobs,
                                                                progress_bar=verbose)

        # get out-of-vocab ensemble names and scores
        ensemble_names_scores[input_out_vocab_ixs] = _get_ensemble_names_scores(ensemble_model,
                                                                                name_freq,
                                                                                input_names[input_out_vocab_ixs],
                                                                                k,
                                                                                None,
                                                                                None,
                                                                                lev_names_scores_input_out_vocab,
                                                                                verbose=verbose)

    return ensemble_names_scores
