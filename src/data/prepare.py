from typing import List, Dict, Tuple
import numpy as np
from editdistance import distance
import re
from unidecode import unidecode

from src.data.constants import surname_prefixes, noise_words, poss_noise_words


def merge_surname_prefixes(name_pieces: List[str]) -> List[str]:
    prefixes = []
    pieces = []
    for piece in name_pieces:
        if piece in surname_prefixes:
            prefixes.append(piece)
        else:
            if len(prefixes) > 0:
                piece = "".join(prefixes) + piece
                prefixes = []
            pieces.append(piece)
    if len(prefixes) > 0:
        piece = "".join(prefixes)
        pieces.append(piece)
    return pieces


def remove_noise_words(name_pieces: List[str]) -> List[str]:
    name_pieces = [piece for piece in name_pieces if piece not in noise_words and len(piece) > 1]
    name_pieces = [piece for piece in name_pieces if piece not in poss_noise_words or len(name_pieces) == 1]
    return name_pieces


def normalize(name: str, is_surname: bool) -> List[str]:
    # remove diacritics
    name = unidecode(name)
    # lowercase
    name = name.lower()
    # replace various forms of apostrophe with empty string
    name = re.sub("[`'´‘’]", "", name)
    # replace all other non-alpha characters with space
    name = re.sub("[^ a-z]", " ", name)
    # replace multiple spaces with a single space and trim
    name = re.sub(" +", " ", name).strip()
    # split into pieces
    pieces = name.split(" ")
    # merge surname prefixes
    pieces = merge_surname_prefixes(pieces)
    # remove noise words
    pieces = remove_noise_words(pieces)
    # return pieces
    return pieces


def levenshtein_similarity(name: str, alt_name: str) -> float:
    return 1.0 - (distance(name, alt_name) / max(len(name), len(alt_name)))


def _get_similarities(name_pieces: List[str], alt_name_pieces: List[str]):
    similarities = np.zeros([len(name_pieces), len(alt_name_pieces)])
    for i, name_piece in enumerate(name_pieces):
        for j, alt_name_piece in enumerate(alt_name_pieces):
            similarities[i, j] = levenshtein_similarity(name_piece, alt_name_piece)
    return similarities


def _get_pairs(name_pieces: List[str], alt_name_pieces: List[str]) -> List[Tuple[str, str]]:
    similarities = _get_similarities(name_pieces, alt_name_pieces)
    pairs = []
    while np.max(similarities) > 0:
        i, j = np.unravel_index(similarities.argmax(), similarities.shape)
        pairs.append((name_pieces[i], alt_name_pieces[j]))
        similarities[i, :] = 0
        similarities[:, j] = 0
    return pairs


def match_name_pairs(row: Dict) -> List[Tuple[str, str]]:
    name_pieces = row["name_pieces"]
    alt_name_pieces = row["alt_name_pieces"]
    if len(name_pieces) == 0 or len(alt_name_pieces) == 0:
        return []
    if len(name_pieces) == 1 and len(alt_name_pieces) == 1:
        return [(name_pieces[0], alt_name_pieces[0])]
    return _get_pairs(name_pieces, alt_name_pieces)
