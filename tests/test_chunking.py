import pytest

from rageval.corpus.chunking import AlignmentError, build_beads, join_sentences, make_chunks, pack_beads
from rageval.corpus.tei import Link, Sentence, TeiDoc


def doc(*sentences):
    return TeiDoc(symbol=None, date=None, sentences=[Sentence(sid, sid.split(":")[0], text) for sid, text in sentences])


EN = doc(("1:1", "One."), ("1:2", "Two."), ("2:1", "Three."))
AR = doc(("1:1", "واحد واثنان."), ("2:1", "ثلاثة."), ("2:2", "زيادة."))
LINKS = [Link(ar=("1:1",), en=("1:1", "1:2")), Link(ar=("2:1",), en=("2:1",)), Link(ar=("2:2",), en=())]


def test_build_beads_follows_links_and_marks_paragraph_ends():
    beads = build_beads(EN, AR, LINKS)
    assert [[s.sid for s in b.en] for b in beads] == [["1:1", "1:2"], ["2:1"], []]
    assert [[s.sid for s in b.ar] for b in beads] == [["1:1"], ["2:1"], ["2:2"]]
    # Bead 1 ends English paragraph 2; bead 2 has no English, so its Arabic sentence decides.
    assert [b.paragraph_end for b in beads] == [True, True, True]
    assert [b.one_to_one for b in beads] == [False, True, False]


def test_paragraph_end_false_mid_paragraph():
    en = doc(("1:1", "A."), ("1:2", "B."))
    ar = doc(("1:1", "أ."), ("1:2", "ب."))
    beads = build_beads(en, ar, [Link(("1:1",), ("1:1",)), Link(("1:2",), ("1:2",))])
    assert [b.paragraph_end for b in beads] == [False, True]


def test_non_monotonic_alignment_is_rejected():
    with pytest.raises(AlignmentError):
        build_beads(EN, AR, [Link(("2:1",), ("2:1",)), Link(("1:1",), ("1:1",))])


def test_missing_sentence_is_rejected():
    with pytest.raises(AlignmentError):
        build_beads(EN, AR, [Link(("9:9",), ("1:1",))])


def test_pack_beads_respects_tighter_language():
    # Arabic reaches the 350 limit first: 150 + 150 fits, a third bead does not.
    assert pack_beads([False] * 3, [100] * 3, [150] * 3, target_tokens=1000, max_tokens=350) == [[0, 1], [2]]


def test_pack_beads_closes_at_paragraph_end_once_target_reached():
    assert pack_beads([True, False, False], [100] * 3, [150] * 3, target_tokens=100, max_tokens=350) == [[0], [1, 2]]


def test_pack_beads_does_not_close_below_target():
    assert pack_beads([True, True], [10, 10], [10, 10], target_tokens=100, max_tokens=350) == [[0, 1]]


def test_oversized_bead_becomes_its_own_chunk():
    assert pack_beads([False, False], [500, 10], [500, 10], target_tokens=100, max_tokens=350) == [[0], [1]]


def test_join_sentences_uses_newline_between_paragraphs():
    assert join_sentences(EN.sentences) == "One. Two.\nThree."
    assert join_sentences([Sentence("1:1", "1", ""), Sentence("1:2", "1", "X.")]) == "X."


def test_make_chunks_keeps_languages_aligned_and_covers_every_bead():
    beads = build_beads(EN, AR, LINKS)
    word_count = lambda texts: [len(t.split()) for t in texts]
    chunks = make_chunks("1992/a/1", beads, word_count, target_tokens=2, max_tokens=3)
    assert [c["chunk_id"] for c in chunks] == ["1992/a/1#0000", "1992/a/1#0001"]
    assert chunks[0]["en"] == "One. Two." and chunks[0]["ar"] == "واحد واثنان."
    assert chunks[1]["en"] == "Three." and chunks[1]["ar"] == "ثلاثة. زيادة."
    assert sum(c["n_beads"] for c in chunks) == len(beads)
    assert chunks[1]["one_to_one_ratio"] == 0.5
