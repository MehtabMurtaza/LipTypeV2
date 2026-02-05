from liptype_rebuild.utils.levenshtein import levenshtein


def test_levenshtein_strings():
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_tokens():
    assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1

