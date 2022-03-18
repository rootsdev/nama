import math

import jellyfish
import numpy as np
# from rapidfuzz.string_metric import levenshtein
from tqdm import tqdm

from src.eval.utils import similars_to_ndarray
from src.models.swivel import get_best_swivel_matches
from src.models.utils import remove_padding


def featurize(swivel_score, lev_score, input_name_freq, candidate_name_freq):
    log_max_freq = math.log10(max(input_name_freq, candidate_name_freq)+1)
    log_min_freq = math.log10(min(input_name_freq, candidate_name_freq)+1)
    return [
        swivel_score,
        swivel_score * log_max_freq,
        swivel_score * log_min_freq,
        lev_score,
        lev_score * log_max_freq,
        lev_score * log_min_freq,
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


def get_best_ensemble_matches(model, vocab, input_names, candidate_names, encoder_model, ensemble_model, name_freq,
                              k, batch_size, add_context=True, n_jobs=1, swivel_threshold=0.45, lev_threshold=0.55):
    swivel_names_scores = get_best_swivel_matches(model=model,
                                                  vocab=vocab,
                                                  input_names=input_names,
                                                  candidate_names=candidate_names,
                                                  encoder_model=encoder_model,
                                                  k=k,
                                                  batch_size=batch_size,
                                                  add_context=add_context,
                                                  n_jobs=n_jobs)
    similar_names_scores = []
    max_similars = 0
    empty_similars = 0
    for input_name, swivels in tqdm(zip(input_names, swivel_names_scores), total=len(input_names)):
        swivel_scores = {name: score for name, score in swivels if score >= swivel_threshold}
        swivel_names = set(swivel_scores.keys())

        # calc lev scores
        lev_scores = {name: score for name, score in
                      _get_lev_similars_for_name(input_name, np.array(list(swivel_names))) if score >= lev_threshold}
        lev_names = set(lev_scores.keys())

        # generate features from swivel and levenshtein scores and frequency
        input_name_freq = name_freq.get(input_name, 0)
        candidate_names = swivel_names.intersection(lev_names)
        features = []
        for candidate_name in candidate_names:
            swivel_score = swivel_scores[candidate_name]
            lev_score = lev_scores[candidate_name]
            candidate_name_freq = name_freq.get(candidate_name, 0)
            features.append(featurize(swivel_score, lev_score, input_name_freq, candidate_name_freq))

        # predict
        if len(features) == 0:
            empty_similars += 1
            predictions = []
        else:
            predictions = ensemble_model.predict_proba(features)[:, 1]
        similar_names_scores.append(list(zip(list(candidate_names), predictions)))
        if len(predictions) > max_similars:
            max_similars = len(predictions)

    # ensure all lists have the same length
    for similars in similar_names_scores:
        if len(similars) < max_similars:
            similars.extend((("", 0.0),)*(max_similars - len(similars)))

    print("max_similars", max_similars)
    print("empty_similars", empty_similars)
    return similars_to_ndarray(similar_names_scores)
