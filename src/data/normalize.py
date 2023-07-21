from typing import List, Set
import regex
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
from src.models import utils


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


def normalize(
        name: str, is_surname: bool, preserve_wildcards: bool = False,
        handle_patronymics: bool = True, dont_return_empty: bool = True
) -> List[str]:
    # remove diacritics and transliterate
    normalized = unidecode(name)
    # lowercase
    normalized = normalized.lower()
    # remove possessive
    normalized = regex.sub("'s$", "", normalized)
    # replace various forms of apostrophe with empty string
    normalized = regex.sub("[`'´‘’]", "", normalized)
    # replace all non-alphanumeric characters with space
    expr = r'[^ \p{L}0-9*?]' if preserve_wildcards else r'[^ \p{L}0-9]'
    normalized = regex.sub(expr, " ", normalized)
    # replace all non-latin-numeric characters with space
    expr = r'[^ a-z0-9*?]' if preserve_wildcards else r'[^ a-z0-9]'
    latin = regex.sub(expr, " ", normalized)
    # replace multiple spaces with a single space and trim
    latin = regex.sub(" +", " ", latin).strip()
    # split into pieces
    pieces = latin.split(" ")
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
    pieces = [regex.sub("[0-9]", "", piece) for piece in pieces]
    # remove noise words again so we remove things like mr1
    pieces = remove_noise_words(pieces, is_surname)
    # remove empty names and single-character surnames
    pieces = [piece for piece in pieces if piece and (len(piece) > 1 or not is_surname)]
    # if no pieces, return the normalized name (or the original name if normalized is empty) with spaces removed
    # this unfortunately creates a loophole that allows names that are not a-z, which have to be removed later
    if dont_return_empty and len(pieces) == 0:
        pieces = [regex.sub("\\s", "", normalized if normalized else name)]
    # standardize patronymics
    if is_surname and handle_patronymics:
        pieces = [standardize_patronymics(piece) for piece in pieces]
    # return pieces
    return pieces


def normalize_freq_names(freq_df, is_surname, add_padding, dont_return_empty=True):
    name_freq = {}
    for name, freq in zip(freq_df["name"], freq_df["frequency"]):
        pieces = normalize(name, is_surname=is_surname, dont_return_empty=dont_return_empty)
        if len(pieces) != 1 or len(pieces[0]) <= 1:
            continue
        name = pieces[0]
        if regex.search("[^a-z]", name):
            continue
        if add_padding:
            name = utils.add_padding(name)
        if name not in name_freq:
            name_freq[name] = freq
    return name_freq
