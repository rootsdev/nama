from src.data.prepare import (
    merge_surname_prefixes,
    remove_noise_words,
    match_name_pairs,
    levenshtein_similarity,
)


test_merge_surname_prefixes_data = [
    {"input": ["mendoza", "y", "gutierres"], "result": ["mendoza", "y", "gutierres"]},
    {"input": ["mendoza", "de", "la", "gutierres"], "result": ["mendoza", "delagutierres"]},
    {"input": ["della", "mendoza", "gutierres"], "result": ["dellamendoza", "gutierres"]},
    {"input": ["mendoza", "gutierres", "de", "la"], "result": ["mendoza", "gutierres", "dela"]},
    {"input": ["van", "der", "leek"], "result": ["vanderleek"]},
    {"input": ["vander", "leek"], "result": ["vanderleek"]},
]


def test_merge_surname_prefixes():
    for test_data in test_merge_surname_prefixes_data:
        result = merge_surname_prefixes(test_data["input"])
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"


test_remove_noise_words_data = [
    {"input": ["mendoza", "y", "gutierres"], "result": ["mendoza", "gutierres"]},
    {"input": ["major", "mendoza"], "result": ["mendoza"]},
    {"input": ["major"], "result": ["major"]},
    {"input": ["d", "r", "gutierres"], "result": ["gutierres"]},
]


def test_remove_noise_words():
    for test_data in test_remove_noise_words_data:
        result = remove_noise_words(test_data["input"])
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"


test_match_name_pairs_data = [
    {
        "input": {"name_pieces": ["john", "smith"], "alt_name_pieces": ["jan", "smythe", "brown"]},
        "result": [("smith", "smythe"), ("john", "jan")],
    }
]


def test_match_name_pairs():
    for test_data in test_match_name_pairs_data:
        result = match_name_pairs(test_data["input"])
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"


test_levenshtein_similarity_data = [
    {"name1": "john", "name2": "jan", "result": 0.5},
    {"name1": "smith", "name2": "smyth", "result": 0.8},
]


def test_levenshtein_similarity():
    for test_data in test_levenshtein_similarity_data:
        result = levenshtein_similarity(test_data["name1"], test_data["name2"])
        assert (
            result == test_data["result"]
        ), f"unexpected result {result} for {test_data['name1']}, {test_data['name2']}"
