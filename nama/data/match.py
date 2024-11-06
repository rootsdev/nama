import numpy as np
from jellyfish import levenshtein_distance


def levenshtein_similarity(name: str, alt_name: str) -> float:
    return 1.0 - (levenshtein_distance(name, alt_name) / max(len(name), len(alt_name)))


def _get_similarities(name_pieces: list[str], alt_name_pieces: list[str]):
    similarities = np.zeros([len(name_pieces), len(alt_name_pieces)])
    for i, name_piece in enumerate(name_pieces):
        for j, alt_name_piece in enumerate(alt_name_pieces):
            similarities[i, j] = levenshtein_similarity(name_piece, alt_name_piece)
    return similarities


def _get_pairs(name_pieces: list[str], alt_name_pieces: list[str]) -> list[tuple[str, str]]:
    similarities = _get_similarities(name_pieces, alt_name_pieces)
    pairs = []
    while np.max(similarities) > 0:
        i, j = np.unravel_index(similarities.argmax(), similarities.shape)
        pairs.append((name_pieces[i], alt_name_pieces[j]))
        similarities[i, :] = 0
        similarities[:, j] = 0
    return pairs


def match_name_pairs(row: dict) -> list[tuple[str, str]]:
    name_pieces = row["name_pieces"]
    alt_name_pieces = row["alt_name_pieces"]
    if len(name_pieces) == 0 or len(alt_name_pieces) == 0:
        return []
    if len(name_pieces) == 1 and len(alt_name_pieces) == 1:
        return [(name_pieces[0], alt_name_pieces[0])]
    return _get_pairs(name_pieces, alt_name_pieces)
