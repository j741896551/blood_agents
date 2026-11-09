"""RAG 可选向量后端：默认零下载，显式启用才建索引。"""

import importlib.util
import subprocess
import sys

import pytest

from blood_agents.knowledge import adapters
from blood_agents.knowledge.rag import Document, HybridRetriever, default_retriever


def test_default_backend_does_not_construct_bge(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(adapters, "BGEEncoder", lambda _: pytest.fail("默认模式加载了大模型"))
    retriever = default_retriever()
    assert retriever.dense_search is None
    assert retriever.search("献血前可以空腹吗")


def test_faiss_factory_injects_real_corpus_and_respects_config(monkeypatch):
    captured = {}

    class Encoder:
        def __init__(self, model_name):
            captured["model_name"] = model_name

    class Store:
        def __init__(self, documents, encoder):
            captured["documents"] = documents
            captured["encoder"] = encoder

        def search(self, query, top_k):
            captured["query"] = query
            captured["top_k"] = top_k
            return [(captured["documents"][0], 0.8)]

    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-bge")
    monkeypatch.setenv("RAG_DENSE_MIN_SCORE", "0.7")
    monkeypatch.setattr(adapters, "BGEEncoder", Encoder)
    monkeypatch.setattr(adapters, "FAISSVectorStore", Store)
    retriever = default_retriever()
    assert captured["model_name"] == "test-bge"
    assert len(captured["documents"]) >= 6
    assert retriever.dense_min_score == 0.7
    assert retriever.search("完全不同的词语", top_k=3)[0]["retrieval_backend"] == "faiss"
    assert captured["query"] == "完全不同的词语"
    assert captured["top_k"] == 6


def test_dense_nearest_neighbor_below_absolute_threshold_is_not_evidence():
    doc = Document("one", "献血前准备", "https://example.org", {"title": "测试"})
    retriever = HybridRetriever(
        [doc], dense_search=lambda _query, _top_k: [(doc, 0.64)],
        min_score=0.15, dense_min_score=0.65,
    )
    assert retriever.search("地铁末班车") == []
    retriever.dense_search = lambda _query, _top_k: [(doc, 0.8)]
    assert retriever.search("地铁末班车")[0]["source"] == "https://example.org"


def test_unknown_backend_and_bad_dense_threshold_fail_explicitly(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "unknown")
    with pytest.raises(ValueError, match="不支持"):
        default_retriever()
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    monkeypatch.setenv("RAG_DENSE_MIN_SCORE", "1.5")
    with pytest.raises(ValueError, match="0 到 1"):
        default_retriever()


def test_bge_query_instruction_only_applies_to_queries():
    class Model:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return texts

    encoder = adapters.BGEEncoder.__new__(adapters.BGEEncoder)
    encoder.model = Model()
    assert encoder.encode(["献血前准备"]) == ["献血前准备"]
    assert encoder.encode_query(["能否空腹献血"])[0].startswith("为这个句子生成表示")


def test_faiss_index_add_and_search_with_fake_embeddings():
    if importlib.util.find_spec("faiss") is None:
        pytest.skip("faiss-cpu 可选依赖未安装")
    # 其他测试会加载 Gradio/Pandas/Torch 的本地扩展；在 macOS 上同进程的
    # FAISS/OpenMP 调用可触发原生层 abort。独立子进程既验证真实 add/search，
    # 也避免把整个回归进程崩掉；线上图进程另有真实模型烟测。
    program = """
import numpy as np
from blood_agents.knowledge.adapters import FAISSVectorStore
from blood_agents.knowledge.rag import Document
docs = [Document("a", "献血前准备", "https://example.org/a", {}),
        Document("b", "收货地址", "demo://blood-station/b", {})]
class Encoder:
    def encode(self, texts):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    def encode_query(self, texts):
        return np.array([[0.9, 0.1]], dtype="float32")
store = FAISSVectorStore(docs, Encoder())
assert store.index.ntotal == 2
assert store.search("献血", top_k=1)[0][0].doc_id == "a"
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
