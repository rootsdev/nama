from typing import List, Dict, Tuple, Set
import numpy as np
from editdistance import distance
import re
from unidecode import unidecode

from src.data.constants import (
    BEGINNING_SURNAME_PREFIXES,
    SURNAME_PREFIXES,
    BEGINNING_GIVEN_PREFIXES,
    GIVEN_PREFIXES,
    GIVEN_ABBREVS,
    NOISE_WORDS,
    POSS_NOISE_WORDS,
    POSS_SURNAME_NOISE_WORDS,
    PATRONYMIC_PATTERNS,
)


def merge_surname_prefixes(name_pieces: List[str]) -> List[str]:
    return _merge_prefixes(BEGINNING_SURNAME_PREFIXES, SURNAME_PREFIXES, name_pieces)


def merge_given_prefixes(name_pieces: List[str]) -> List[str]:
    return _merge_prefixes(BEGINNING_GIVEN_PREFIXES, GIVEN_PREFIXES, name_pieces)


def _merge_prefixes(beginning_prefixes: Set[str], anywhere_prefixes: Set[str], name_pieces: List[str]) -> List[str]:
    prefixes = []
    pieces = []
    for ix, piece in enumerate(name_pieces):
        if (ix == 0 and piece in beginning_prefixes) or (piece in anywhere_prefixes):
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


def expand_given_abbrevs(name_pieces: List[str]) -> List[str]:
    return [GIVEN_ABBREVS.get(piece, piece) for piece in name_pieces]


def remove_noise_words(name_pieces: List[str], is_surname: bool) -> List[str]:
    name_pieces = [piece for piece in name_pieces if piece not in NOISE_WORDS]
    # keep the last possible noise word if its the only word left
    if len(name_pieces) > 1:
        result = [
            piece
            for piece in name_pieces
            if piece not in POSS_NOISE_WORDS and not (is_surname and piece in POSS_SURNAME_NOISE_WORDS)
        ]
        name_pieces = result if len(result) > 0 else [name_pieces[-1]]

    return name_pieces


def standardize_patronymics(name: str) -> str:
    for patronymic_pattern in PATRONYMIC_PATTERNS:
        name = patronymic_pattern["pattern"].sub(patronymic_pattern["replacement"], name)
    return name


def normalize(name: str, is_surname: bool, preserve_wildcards: bool = False) -> List[str]:
    # remove diacritics
    normalized = unidecode(name)
    # lowercase
    normalized = normalized.lower()
    # remove possessive
    normalized = re.sub("'s$", "", normalized)
    # replace various forms of apostrophe with empty string
    normalized = re.sub("[`'´‘’]", "", normalized)
    # replace all other non-alphanumeric characters with space
    regex = "[^ a-z0-9*?]" if preserve_wildcards else "[^ a-z0-9]"
    normalized = re.sub(regex, " ", normalized)
    # replace multiple spaces with a single space and trim
    normalized = re.sub(" +", " ", normalized).strip()
    # split into pieces
    pieces = normalized.split(" ")
    # expand abbrevs
    if not is_surname:
        pieces = expand_given_abbrevs(pieces)
    # merge prefixes
    if is_surname:
        pieces = merge_surname_prefixes(pieces)
    else:
        pieces = merge_given_prefixes(pieces)
    # remove noise words
    pieces = remove_noise_words(pieces, is_surname)
    # remove numbers (kept until now so we could remove 1st as a noise word instead of having st as a prefix)
    pieces = [re.sub("[0-9]", "", piece) for piece in pieces]
    # TODO consider removing noise words again so we remove things like mr1
    # remove empty names and single-character surnames
    pieces = [piece for piece in pieces if piece and (len(piece) > 1 or not is_surname)]
    # if no pieces, return the normalized name (or the original name if normalized is empty) with spaces removed
    if len(pieces) == 0:
        pieces = [re.sub("\\s", "", normalized if normalized else name)]
    # standardize patronymics
    if is_surname:
        pieces = [standardize_patronymics(piece) for piece in pieces]
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
