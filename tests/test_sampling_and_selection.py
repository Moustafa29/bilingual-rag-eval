from itertools import islice

from rageval.corpus.unpc import PairStats, cited_closure, is_eligible, select_seeds, selection_key
from rageval.questions.sampling import bridge_candidates, group_by_doc, is_content_chunk, single_candidates

RULES = {"min_chars": 100, "min_one_to_one_ratio": 0.8, "min_letter_ratio": 0.6, "max_toc_line_ratio": 0.3, "min_paragraph_words": 12}
PROSE = (
    "The General Assembly requested the Secretary-General to submit a report on the implementation "
    "of the programme of action at its next session, including recommendations on financing."
)


def chunk(chunk_id, en=PROSE, ar="نص عربي", ratio=1.0, truncated=False):
    return {"chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0], "en": en, "ar": ar, "one_to_one_ratio": ratio, "truncated_e5": truncated}


def test_content_filter():
    assert is_content_chunk(chunk("d#0"), RULES)
    toc = "\n".join(f"Chapter {i} on important matters of the programme . {i + 2}" for i in range(6))
    assert not is_content_chunk(chunk("d#0", en=toc), RULES)
    assert not is_content_chunk(chunk("d#0", en="Distr. GENERAL A/47/10"), RULES)
    assert not is_content_chunk(chunk("d#0", ratio=0.5), RULES)
    assert not is_content_chunk(chunk("d#0", truncated=True), RULES)
    assert not is_content_chunk(chunk("d#0", ar=" "), RULES)


def test_eligibility():
    rules = {"min_one_to_one_links": 10, "max_one_to_one_links": 100, "min_one_to_one_ratio": 0.8}
    assert is_eligible(PairStats("d", 0.4, n_links=50, n_one_to_one=45), rules)
    assert not is_eligible(PairStats("d", 0.4, n_links=50, n_one_to_one=30), rules)
    assert not is_eligible(PairStats("d", 0.4, n_links=5, n_one_to_one=5), rules)
    assert not is_eligible(PairStats("d", 0.4, n_links=500, n_one_to_one=500), rules)


def test_seed_selection_is_hash_ordered_and_stable_under_additions():
    ids = [f"2001/a/{i}" for i in range(50)]
    seeds = select_seeds(ids, "salt", 10)
    assert seeds == sorted(ids, key=lambda d: selection_key("salt", d))[:10]
    # Adding a document either enters the top 10 or leaves the selection unchanged; it never reorders it.
    with_extra = select_seeds(ids + ["2014/s/999"], "salt", 10)
    assert with_extra == seeds or with_extra[:-1] == [s for s in seeds if s in with_extra][: len(with_extra) - 1]
    assert select_seeds(ids, "other-salt", 10) != seeds


def test_cited_closure_excludes_seeds_and_caps():
    key_to_docs = {"a/1": ["1990/a/1"], "a/2": ["1990/a/2"], "a/3": ["1990/a/3"], "s/9": ["1990/s/9"]}
    cited = {"1990/s/9": {"a/1", "a/2", "a/3", "s/9"}}
    added = cited_closure(["1990/s/9"], cited, key_to_docs, "salt", max_added=2)
    assert len(added) == 2 and "1990/s/9" not in added
    assert added == sorted(["1990/a/1", "1990/a/2", "1990/a/3"], key=lambda d: selection_key("salt", d))[:2]


def test_single_candidates_one_per_doc_and_deterministic():
    chunks = [chunk(f"199{d}/a/{d}#000{k}") for d in range(5) for k in range(3)]
    by_doc = group_by_doc(chunks)
    first = [c["chunk_id"] for c in single_candidates(by_doc, RULES, seed=1)]
    assert first == [c["chunk_id"] for c in single_candidates(by_doc, RULES, seed=1)]
    assert len({cid.split("#")[0] for cid in first}) == len(first) == 5


def test_bridge_candidates_link_citing_chunk_to_cited_document():
    citing = chunk("2001/a/56/1#0000", en=PROSE + " See document A/55/7.")
    cited = chunk("2000/a/55/7#0000")
    by_doc = group_by_doc([citing, cited])
    key_to_docs = {"a/56/1": ["2001/a/56/1"], "a/55/7": ["2000/a/55/7"]}
    found = list(islice(bridge_candidates(by_doc, key_to_docs, RULES, seed=1), 5))
    assert found == [(citing, cited, "A/55/7")]
