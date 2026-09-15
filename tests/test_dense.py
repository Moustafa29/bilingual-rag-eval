import numpy as np

from rageval.retrieval.dense import MODELS, Encoder, exact_search, texts_fingerprint


def test_exact_search_returns_best_inner_products_in_order():
    passages = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    queries = np.array([[0, 1], [1, 0]], dtype=np.float32)
    ids, scores = exact_search(passages, queries, k=2)
    assert ids.tolist() == [[1, 2], [0, 2]]
    assert np.allclose(scores, [[1.0, 0.8], [1.0, 0.6]])


def test_e5_uses_asymmetric_prefixes_and_bge_m3_none():
    assert (MODELS["e5-base"].query_prefix, MODELS["e5-base"].passage_prefix) == ("query: ", "passage: ")
    assert (MODELS["bge-m3"].query_prefix, MODELS["bge-m3"].passage_prefix) == ("", "")


def test_fingerprint_changes_with_prefix_text_and_model():
    base = texts_fingerprint(["a"], "query: ", MODELS["e5-base"])
    assert base != texts_fingerprint(["a"], "passage: ", MODELS["e5-base"])
    assert base != texts_fingerprint(["b"], "query: ", MODELS["e5-base"])
    assert base != texts_fingerprint(["a"], "query: ", MODELS["bge-m3"])


class FakeST:
    def __init__(self):
        self.calls = []

    def encode(self, texts, prompt=None, **kwargs):
        self.calls.append((list(texts), prompt, kwargs.get("normalize_embeddings")))
        return np.ones((len(texts), 3), dtype=np.float32)


def test_encoder_passes_prefix_normalizes_and_caches(tmp_path):
    encoder = Encoder(MODELS["e5-base"], tmp_path)
    fake = FakeST()
    encoder._st = fake
    first = encoder.encode(["who?"], "query")
    second = encoder.encode(["who?"], "query")
    assert fake.calls == [(["who?"], "query: ", True)]
    assert np.array_equal(first, second)
    encoder.encode(["text"], "passage")
    assert fake.calls[-1][1] == "passage: "
