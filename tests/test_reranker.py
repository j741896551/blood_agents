"""精排只改变已有证据的顺序，不改变拒答门槛或片段来源。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.knowledge import adapters
from blood_agents.knowledge.rag import Document, HybridRetriever, default_retriever
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id, text, f"https://example.org/{doc_id}", {"title": doc_id})


def test_reranker_reorders_only_eligible_hits_and_keeps_retrieval_scores():
    first = _doc("first", "献血前准备")
    second = _doc("second", "献血后休息")
    weak = _doc("weak", "地铁末班车")
    seen = {}

    def rerank(_query, candidates):
        seen["ids"] = [doc.doc_id for doc, _ in candidates]
        # 原始相似度较低的 second 被精排提升到第一，负 logit 也可以排序。
        return [(second, -2.0), (first, -5.0)]

    retriever = HybridRetriever(
        [first, second, weak],
        dense_search=lambda _query, _top_k: [(first, 0.8), (second, 0.6), (weak, 0.14)],
        reranker=rerank, min_score=0.15, dense_min_score=0.0, backend="faiss",
    )
    hits = retriever.search("无重合词", top_k=3)
    assert seen["ids"] == ["first", "second"]
    assert [hit["doc_id"] for hit in hits] == ["second", "first"]
    assert [hit["score"] for hit in hits] == [0.6, 0.8]
    assert [hit["rerank_score"] for hit in hits] == [-2.0, -5.0]
    assert hits[0]["source"] == second.source
    assert retriever.search("无重合词", top_k=0) == []


def test_reranker_is_not_invoked_without_evidence():
    doc = _doc("one", "献血前准备")
    retriever = HybridRetriever(
        [doc], dense_search=lambda _query, _top_k: [(doc, 0.14)],
        reranker=lambda *_: pytest.fail("无证据却调用了精排模型"),
        min_score=0.15, dense_min_score=0.0,
    )
    assert retriever.search("地铁末班车") == []


@pytest.mark.parametrize("output", [
    lambda doc: [(doc, float("nan"))],
    lambda doc: [(_doc("unknown", "伪造片段"), 1.0)],
    lambda _doc: [],
])
def test_invalid_rerank_output_fails_closed(output):
    doc = _doc("one", "献血前准备")
    retriever = HybridRetriever(
        [doc], reranker=lambda _query, _candidates: output(doc), min_score=0.15,
    )
    with pytest.raises(ValueError, match="精排结果"):
        retriever.search("献血")


def test_bge_reranker_pairs_query_with_each_document_and_sorts_raw_scores():
    first = _doc("first", "献血前准备")
    second = _doc("second", "献血后休息")

    class Model:
        def predict(self, pairs):
            assert pairs == [("献血", first.text), ("献血", second.text)]
            return [-4.0, 2.0]

    reranker = adapters.BGEReranker.__new__(adapters.BGEReranker)
    reranker.model = Model()
    assert reranker("献血", []) == []
    assert reranker("献血", [(first, 0.8), (second, 0.6)]) == [
        (second, 2.0), (first, -4.0),
    ]


def test_reranker_factory_requires_explicit_switch(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    monkeypatch.setattr(adapters, "BGEReranker", lambda _: pytest.fail("默认加载了精排模型"))
    assert default_retriever().reranker is None

    captured = {}

    class Reranker:
        def __init__(self, model_name):
            captured["model"] = model_name

        def __call__(self, _query, candidates):
            return [(doc, 1.0) for doc, _ in candidates]

    monkeypatch.setattr(adapters, "BGEReranker", Reranker)
    monkeypatch.setenv("RAG_RERANKER", "bge")
    monkeypatch.setenv("RERANKER_MODEL", "test-bge-reranker")
    retriever = default_retriever()
    assert captured["model"] == "test-bge-reranker"
    assert retriever.search("献血前可以空腹吗")[0]["rerank_score"] == 1.0

    monkeypatch.setenv("RAG_RERANKER", "unknown")
    with pytest.raises(ValueError, match="RAG_RERANKER"):
        default_retriever()


def test_faq_graph_passes_separate_rerank_score_to_reply_model(monkeypatch):
    class Reranker:
        def __init__(self, _model_name):
            pass

        def __call__(self, _query, candidates):
            return [(doc, -2.0) for doc, _ in candidates]

    class Navigation:
        def invoke(self, _messages):
            return AIMessage(content='{"intent":"faq"}')

    class Reply:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            hits = facts["retrieved_context"]
            assert hits and hits[0]["rerank_score"] == -2.0
            assert hits[0]["score"] > 0
            sources = list(dict.fromkeys(hit["source"] for hit in hits))
            disclosure = "引用的 demo:// 资料均为虚构演示。" if any(
                hit["source_kind"] == "synthetic" for hit in hits) else ""
            return AIMessage(content=f"献血前请勿空腹。{disclosure}来源：{'、'.join(sources)}")

    monkeypatch.setenv("VECTOR_BACKEND", "memory")
    monkeypatch.setenv("RAG_RERANKER", "bge")
    monkeypatch.setattr(adapters, "BGEReranker", Reranker)
    graph = build_graph(
        BloodRepository(), IntentRouter(llm=Navigation(), response_llm=Reply(), risk_llm=False,
                                        evidence_llm=False),
        checkpointer=MemorySaver(),
    )
    state = graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content="献血前可以空腹吗？")]},
        {"configurable": {"thread_id": "rerank-faq-test"}},
    )
    assert state["result"]["status"] == "grounded"
    assert state["response_info"]["mode"] == "llm"
