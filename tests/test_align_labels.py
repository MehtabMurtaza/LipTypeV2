from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.labels import Charset


def test_labels_roundtrip_basic():
    cs = Charset()
    text = "set red in t zero now"
    labels = cs.text_to_labels(text)
    assert all(isinstance(x, int) for x in labels)
    assert cs.labels_to_text(labels) == text


def test_parse_align_and_sentence(tmp_path):
    p = tmp_path / "x.align"
    p.write_text("0 1000 sil\n1000 2000 set\n2000 3000 sp\n3000 4000 red\n", encoding="utf-8")
    items = parse_align_file(str(p))
    assert len(items) == 4
    sent = align_to_sentence(items)
    assert sent == "set red"

