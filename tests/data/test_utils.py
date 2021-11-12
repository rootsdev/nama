import pandas as pd

from src.data.utils import _load


def test_load():
    df = pd.DataFrame(
        {
            "name1": ["john", "mary", "john"],
            "name2": ["johnny", "marie", "jonathan"],
            "weighted_count": [0.7, 1.0, 0.3],
            "co_occurrence": [7, 1, 3],
        }
    )
    input_names, weighted_actual_names, candidate_names = _load(df)

    assert len(input_names) == 2, "should have two input names"
    assert input_names[0] == "john", "first input name should be john"
    assert len(candidate_names) == 3, "should have three candidate names"
    assert len(weighted_actual_names[0]) == 3, "john should have three weighted actual names"
    assert len(weighted_actual_names[1]) == 2, "mary should have two weighted actual names"
    assert weighted_actual_names[0][0] == ("johnny", 0.7, 7)
    assert weighted_actual_names[0][1] == ("jonathan", 0.3, 3)
    assert weighted_actual_names[0][2] == ("john", 0.0, 0)
    assert weighted_actual_names[1][0] == ("marie", 1.0, 1)
    assert weighted_actual_names[1][1] == ("mary", 0.0, 0)
