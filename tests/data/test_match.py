from src.data.match import (
    match_name_pairs,
    levenshtein_similarity,
)


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
