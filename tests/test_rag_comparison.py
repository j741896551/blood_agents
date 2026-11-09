"""RAG 挑战集与精排对比报告的可重复结构验证。"""

import json
from pathlib import Path

import pytest

from blood_agents.evals.rag_comparison import (
    compare,
    dataset_identity,
    evaluate_cases,
    load_cases,
)
from blood_agents.knowledge.rag import Document, HybridRetriever


def test_challenge_dataset_covers_all_sources_and_hard_negatives():
    path = Path(__file__).parents[1] / "evals" / "rag_challenge.json"
    cases = load_cases(path)
    assert len(cases) == 28
    positives = [case for case in cases if case["sources"]]
    negatives = [case for case in cases if not case["sources"]]
    assert len(positives) == 15 and len(negatives) == 13
    assert len({source for case in positives for source in case["sources"]}) == 10
    assert sum(case["group"] == "unsupported_static" for case in negatives) == 7
    assert not any(source.startswith("demo://") for case in cases for source in case["sources"])


def test_dataset_identity_changes_when_question_changes(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('{"revision":"v1","cases":[]}', encoding="utf-8")
    before = dataset_identity(path)
    path.write_text('{"revision":"v2","cases":[]}', encoding="utf-8")
    after = dataset_identity(path)
    assert before["dataset_revision"] == "v1"
    assert after["dataset_revision"] == "v2"
    assert before["dataset_sha256"] != after["dataset_sha256"]


def test_challenge_dataset_rejects_duplicate_ids_and_unknown_source(tmp_path):
    path = tmp_path / "challenge.json"
    case = {"id": "one", "group": "test", "query": "测试", "sources": []}
    path.write_text(json.dumps({"cases": [case, case]}), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_cases(path)
    path.write_text(json.dumps({"cases": [{**case, "sources": ["https://unknown.example"]}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="白名单"):
        load_cases(path)


def test_detailed_metrics_keep_positive_and_no_evidence_separate():
    cases = [
        {"id": "P", "group": "positive", "query": "正例", "sources": ["https://example.org/a"]},
        {"id": "N", "group": "negative", "query": "负例", "sources": []},
    ]

    class Retriever:
        def search(self, query, top_k):
            assert top_k == 4
            if query == "正例":
                return [{"source": "https://example.org/a", "doc_id": "a", "score": 0.8}]
            return []

    result = evaluate_cases(cases, Retriever())
    assert result["metrics"]["recall@4"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["metrics"]["no_evidence_rate"] == 1.0
    assert result["metrics"]["pass_by_group"]["negative"] == {"passed": 1, "count": 1}
    assert result["metrics"]["query_latency_p95_ms"] >= 0


def test_comparison_shares_base_candidates_and_exposes_changed_top1():
    first = Document("first", "献血前", "https://example.org/a", {})
    second = Document("second", "献血后", "https://example.org/b", {})
    base = HybridRetriever([first, second], min_score=0.01, backend="memory")

    def reranker(_query, candidates):
        return [(doc, float(index)) for index, (doc, _) in
                enumerate(reversed(candidates), start=1)]

    cases = [{"id": "P", "group": "positive", "query": "献血", "sources": [second.source]}]
    report = compare(cases, base, reranker, top_k=2)
    assert report["changed_top1"] == ["P"]
    assert report["baseline"]["cases"][0]["top_doc_id"] == "first"
    assert report["reranked"]["cases"][0]["top_doc_id"] == "second"
    assert report["reranked"]["cases"][0]["top_rerank_score"] == 1.0
