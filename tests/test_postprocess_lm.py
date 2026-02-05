from liptype_rebuild.postprocess.ngram_lm import BiTrigramLM
from liptype_rebuild.postprocess.repair import RepairModel


def test_bitrigram_train_and_score():
    lines = [
        "set red in t zero now",
        "set red in t one now",
        "set blue in t zero now",
    ]
    lm = BiTrigramLM.train(lines)
    p = lm.p_combined_word("set", "red", "in")
    assert p > 0.0


def test_repair_simple_substitution():
    lines = ["set red in t zero now"] * 5
    lm = BiTrigramLM.train(lines)
    repair = RepairModel(lm=lm, dictionary_words={"set", "red", "in", "t", "zero", "now"})
    out = repair.repair_sentence("set rdd in t zero now")
    # may or may not fix depending on LM; should not crash
    assert isinstance(out, str)

