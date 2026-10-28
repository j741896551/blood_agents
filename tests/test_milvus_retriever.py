"""Milvus 显式后端的索引生命周期与查询契约。"""

import sys
import types

import numpy as np
import pytest

from blood_agents.knowledge import adapters
from blood_agents.knowledge.rag import Document, default_retriever


def test_milvus_factory_is_explicit_and_passes_configuration(monkeypatch):
    captured = {}

    class Encoder:
        def __init__(self, model_name):
            captured["model"] = model_name

    class Store:
        def __init__(self, documents, encoder, **kwargs):
            captured.update(kwargs)
            captured["documents"] = documents

        def search(self, _query, _top_k):
            return [(captured["documents"][0], 0.9)]

    monkeypatch.setattr(adapters, "BGEEncoder", Encoder)
    monkeypatch.setattr(adapters, "MilvusVectorStore", Store)
    monkeypatch.setenv("VECTOR_BACKEND", "milvus")
    monkeypatch.setenv("RAG_MILVUS_URI", "/tmp/test-milvus.db")
    monkeypatch.setenv("MILVUS_COLLECTION", "demo_blood")
    monkeypatch.setenv("MILVUS_TOKEN", "test-token")
    retriever = default_retriever()
    assert captured["model"] == "BAAI/bge-large-zh-v1.5"
    assert captured["uri"] == "/tmp/test-milvus.db"
    assert captured["collection_name"] == "demo_blood"
    assert captured["token"] == "test-token"
    assert retriever.search("完全不同词语")[0]["retrieval_backend"] == "milvus"


def test_milvus_reuses_collection_and_only_fills_missing_vectors(monkeypatch, tmp_path):
    collections = {}
    writes = []

    class Client:
        def __init__(self, uri, **kwargs):
            assert uri == str(tmp_path / "knowledge.db")
            assert kwargs == {}

        def has_collection(self, collection_name):
            return collection_name in collections

        def load_collection(self, collection_name):
            assert collection_name in collections

        def create_collection(self, collection_name, dimension, metric_type, auto_id):
            assert metric_type == "IP" and auto_id is False
            collections[collection_name] = {"dim": dimension, "rows": {}}

        def get(self, collection_name, ids, output_fields):
            assert output_fields == ["id"]
            return [{"id": index} for index in ids if index in collections[collection_name]["rows"]]

        def describe_collection(self, collection_name):
            return {"fields": [{"name": "vector", "params": {
                "dim": collections[collection_name]["dim"],
            }}]}

        def insert(self, collection_name, data):
            writes.append((collection_name, [row["id"] for row in data]))
            collections[collection_name]["rows"].update({row["id"]: row for row in data})

        def search(self, collection_name, data, limit, output_fields):
            assert output_fields == ["id"]
            query = np.array(data[0])
            rows = collections[collection_name]["rows"].values()
            return [[{"id": row["id"], "distance": float(np.dot(row["vector"], query))}
                     for row in sorted(rows, key=lambda row: -np.dot(row["vector"], query))[:limit]]]

    class Encoder:
        model_name = "test-bge"

        def encode(self, texts):
            return np.array([[1.0, 0.0] if "献血" in text else [0.0, 1.0]
                             for text in texts], dtype="float32")

        def encode_query(self, _texts):
            return np.array([[0.9, 0.1]], dtype="float32")

    monkeypatch.setitem(sys.modules, "pymilvus", types.SimpleNamespace(MilvusClient=Client))
    docs = [Document("a", "献血前准备", "https://example.org/a", {}),
            Document("b", "地址填写", "demo://blood-station/b", {})]
    uri = str(tmp_path / "knowledge.db")
    first = adapters.MilvusVectorStore(docs, Encoder(), uri=uri)
    assert len(writes) == 1 and writes[0][1] == [0, 1]
    second = adapters.MilvusVectorStore(docs, Encoder(), uri=uri)
    assert first.collection_name == second.collection_name
    assert len(writes) == 1  # 重启后只读取，不再次写入相同语料。
    assert second.search("献血", top_k=1)[0][0].doc_id == "a"

    # 模拟中断后缺一个向量，只补缺失 ID，不覆盖已有数据。
    del collections[first.collection_name]["rows"][1]
    adapters.MilvusVectorStore(docs, Encoder(), uri=uri)
    assert writes[-1][1] == [1]

    changed = [docs[0], Document("b", "新的地址说明", docs[1].source, {})]
    third = adapters.MilvusVectorStore(changed, Encoder(), uri=uri)
    assert third.collection_name != first.collection_name
    assert first.collection_name in collections  # 内容变化不删除历史索引。


def test_milvus_rejects_invalid_collection_name_without_connection(monkeypatch):
    docs = [Document("a", "正文", "demo://blood-station/a", {})]
    with pytest.raises(ValueError, match="MILVUS_COLLECTION"):
        adapters.MilvusVectorStore(docs, object(), collection_name="unsafe-name")
