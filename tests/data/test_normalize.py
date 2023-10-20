from src.data.normalize import (
    normalize,
    merge_surname_prefixes,
    remove_noise_words,
    standardize_patronymics,
)


test_normalize_data = [
    {"input": "den edw", "is_surname": False, "result": ["dennis", "edward"]},
    {"input": "den edw", "is_surname": True, "result": ["denedw"]},
    {"input": "constance de la luz", "is_surname": False, "result": ["constance", "delaluz"]},
    {"input": "constance mc adams", "is_surname": False, "result": ["constance", "mc", "adams"]},
    {"input": "john's", "is_surname": False, "result": ["john"]},
    {"input": "1st john", "is_surname": False, "result": ["john"]},
    {"input": "j0hn", "is_surname": False, "result": ["jhn"]},
    {"input": "a'lo`ha", "is_surname": False, "result": ["aloha"]},
    {"input": "ui li", "is_surname": False, "result": ["ui", "li"]},
    {"input": "ui li", "is_surname": True, "result": ["uili"]},
    {"input": "ui", "is_surname": True, "result": ["ui"]},
    {"input": "John Paul", "is_surname": False, "result": ["john", "paul"]},
    {"input": "John-Paul", "is_surname": False, "result": ["john", "paul"]},
    {"input": "Василий", "is_surname": False, "result": ["vasilii"]},
    {"input": "王李", "is_surname": False, "result": ["wang", "li"]},
    {"input": "Смирнов", "is_surname": True, "result": ["smirnov"]},
    {"input": "Quitéria Da Conceição", "is_surname": True, "result": ["quiteria", "daconceicao"]},
    {"input": "Garcia O Ochoa", "is_surname": True, "result": ["garcia", "ochoa"]},
    {"input": "O Ochoa", "is_surname": True, "result": ["oochoa"]},
    {"input": "de Ochoa de Gutierrez", "is_surname": True, "result": ["deochoa", "degutierrez"]},
    {"input": "Sir King", "is_surname": True, "result": ["king"]},
    {"input": "Sir Jones King", "is_surname": True, "result": ["jones"]},
    {"input": "Sir Jones", "is_surname": True, "result": ["jones"]},
    {"input": "D R Jones", "is_surname": True, "result": ["jones"]},
    {"input": "D X Jones", "is_surname": True, "result": ["dx", "jones"]},
    {"input": "d r john", "is_surname": False, "result": ["d", "r", "john"]},
    {"input": "d gutierres", "is_surname": False, "result": ["d", "gutierres"]},
    {"input": "d gutierres", "is_surname": True, "result": ["dgutierres"]},
    {"input": "d gutierres", "is_surname": True, "result": ["dgutierres"]},
    {"input": "mendoza y gutierres", "is_surname": False, "result": ["mendoza", "gutierres"]},
    {"input": "mendoza j gutierres", "is_surname": False, "result": ["mendoza", "j", "gutierres"]},
    {"input": "mendoza y gutierres", "is_surname": True, "result": ["mendoza", "gutierres"]},
    {"input": "J", "is_surname": True, "result": ["j"]},
    {"input": "J", "is_surname": False, "result": ["j"]},
    {"input": "0 1 2 3", "is_surname": False, "result": ["0123"]},
    {"input": "0 1 2 3John@", "is_surname": False, "result": ["john"]},
    {"input": "van der leek", "is_surname": True, "result": ["vanderleek"]},
    {"input": "vander leek", "is_surname": True, "result": ["vanderleek"]},
    {"input": "van derleek", "is_surname": True, "result": ["vanderleek"]},
    {"input": "vanderleek", "is_surname": True, "result": ["vanderleek"]},
    {"input": "van der leek horn", "is_surname": True, "result": ["vanderleek", "horn"]},
    {"input": "Pastor50y", "is_surname": True, "result": ["pastory"]},
    {"input": "Pastor 50 y", "is_surname": True, "result": ["pastor50y"]},
    {"input": "Baby1", "is_surname": False, "result": ["baby1"]},
    {"input": "Mr. Smith", "is_surname": True, "result": ["smith"]},
    {"input": "Mr Smith", "is_surname": True, "result": ["smith"]},
    {"input": "Mr.", "is_surname": True, "result": ["mr"]},
    {"input": "A", "is_surname": True, "result": ["a"]},
    {"input": "A Smith", "is_surname": True, "result": ["asmith"]},
    {"input": "Z Smith", "is_surname": True, "result": ["smith"]},
    {"input": "A", "is_surname": False, "result": ["a"]},
    {"input": "A John", "is_surname": False, "result": ["a", "john"]},
    {"input": "Jo?n* Sm?th", "is_surname": False, "preserve_wildcards": True, "result": ["jo?n*", "sm?th"]},
    {"input": "Jo?n* Sm?th", "is_surname": False, "preserve_wildcards": False, "result": ["jo", "n", "sm"]},
    {"input": "MR 1", "is_surname": False, "preserve_wildcards": False, "result": ["mr1"]},
    {"input": "Z JONES", "is_surname": True, "preserve_wildcards": False, "result": ["jones"]},
    {"input": "MR1 JONES", "is_surname": True, "preserve_wildcards": False, "result": ["jones"]},
    {"input": "da john", "is_surname": False, "preserve_wildcards": False, "result": ["dajohn"]},
    {"input": "wm", "is_surname": False, "preserve_wildcards": False, "result": ["william"]},
    {"input": "dejesus", "is_surname": False, "preserve_wildcards": False, "result": ["dejesus"]},
    {"input": "dejesus", "is_surname": True, "preserve_wildcards": False, "result": ["dejesus"]},
    {"input": "de jesus", "is_surname": False, "preserve_wildcards": False, "result": ["dejesus"]},
    {"input": "de jesus", "is_surname": True, "preserve_wildcards": False, "result": ["dejesus"]},
]


def test_normalize():
    for test_data in test_normalize_data:
        result = normalize(
            test_data["input"], test_data["is_surname"], preserve_wildcards=test_data.get("preserve_wildcards", False)
        )
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"


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
    {"input": ["mendoza", "y", "gutierres"], "is_surname": False, "result": ["mendoza", "gutierres"]},
    {"input": ["major", "mendoza"], "is_surname": False, "result": ["mendoza"]},
    {"input": ["major"], "is_surname": False, "result": ["major"]},
    {"input": ["sir", "smith"], "is_surname": False, "result": ["smith"]},
    {"input": ["sir", "king"], "is_surname": False, "result": ["king"]},
    {"input": ["smith", "king"], "is_surname": False, "result": ["smith"]},
    {"input": ["king", "smith"], "is_surname": False, "result": ["smith"]},
]


def test_remove_noise_words():
    for test_data in test_remove_noise_words_data:
        result = remove_noise_words(test_data["input"], test_data["is_surname"])
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"


test_standardize_patronymics_data = [
    {"input": "hansdr", "result": "hanson"},
    {"input": "petrokovna", "result": "petrokovich"},
    {"input": "radovichna", "result": "radovich"},
    {"input": "radovinichna", "result": "radovich"},
    {"input": "chesworth", "result": "chesworth"},
]


def test_standardiza_patronymics():
    for test_data in test_standardize_patronymics_data:
        result = standardize_patronymics(test_data["input"])
        assert result == test_data["result"], f"unexpected result {result} for {test_data['input']}"
