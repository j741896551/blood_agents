"""FAQ 模型证据核验：命中不等于能回答，失败时不放行片段。"""

import json
import stat

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from blood_agents.evals.rag_evidence_evaluation import evaluate_evidence, save_report, select_cases
from blood_agents.knowledge.evidence import EvidenceAssessor, EvidenceCheckUnavailable
from blood_agents.llm.model_gateway import ModelGateway, ModelProvider
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


class Navigation:
    def invoke(self, _messages):
        return AIMessage(content='{"intent":"faq"}')


class Retriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query):
        del query
        return self.hits


class EvidenceModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke(self, messages):
        self.calls.append(json.loads(messages[-1].content))
        return AIMessage(content=self.output)


class FailingEvidenceModel:
    """只模拟供应商故障，不发起外部模型请求。"""

    def __init__(self, error):
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        raise self.error


class Reply:
    def __init__(self):
        self.facts = None

    def invoke(self, messages):
        self.facts = json.loads(messages[-1].content)
        result = self.facts["result"]
        if result["status"] == "evidence_check_failed":
            return AIMessage(content="证据核验暂不可用，目前无法核实资料是否足够，请稍后重试。")
        if result["status"] == "no_evidence":
            return AIMessage(content="现有资料不足以回答这个具体问题，无法确认所问事实。")
        hit = self.facts["retrieved_context"][0]
        return AIMessage(content=f"片段说明：{hit['text']} 来源：{hit['source']}")


def ask(question, hits, evidence_output):
    evidence = EvidenceModel(evidence_output)
    reply = Reply()
    graph = build_graph(
        BloodRepository(),
        IntentRouter(llm=Navigation(), response_llm=reply, risk_llm=False,
                     evidence_llm=evidence),
        retriever=Retriever(hits), checkpointer=MemorySaver(),
    )
    state = graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "faq-evidence-check"}},
    )
    return state, evidence, reply


def ask_with_evidence_gateway(question, hits, gateway):
    reply = Reply()
    graph = build_graph(
        BloodRepository(),
        IntentRouter(llm=Navigation(), response_llm=reply, risk_llm=False,
                     evidence_llm=gateway),
        retriever=Retriever(hits), checkpointer=MemorySaver(),
    )
    state = graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": "faq-evidence-failover"}},
    )
    return state, reply


def test_evidence_timeout_uses_backup_before_handing_snippet_to_reply():
    hit = {"doc_id": "law", "text": "两次采血间隔不少于六个月。",
           "source": "https://example.org/law"}
    primary = FailingEvidenceModel(TimeoutError("primary unavailable"))
    backup = EvidenceModel('{"supported":true,"supporting_doc_ids":["law"]}')
    gateway = ModelGateway([
        ModelProvider("primary", primary), ModelProvider("backup", backup),
    ], total_timeout=2)

    state, reply = ask_with_evidence_gateway("两次采血最少间隔多久", [hit], gateway)

    assert len(primary.calls) == len(backup.calls) == 1
    assert json.loads(primary.calls[0][-1].content) == backup.calls[0]
    assert state["result"]["status"] == "grounded"
    assert state["result"]["citations"] == [hit["source"]]
    assert reply.facts["retrieved_context"] == [hit]


def test_evidence_both_providers_fail_closed_without_leaking_candidate():
    hit = {"doc_id": "law", "text": "两次采血间隔不少于六个月。",
           "source": "https://example.org/law"}
    primary = FailingEvidenceModel(TimeoutError("primary unavailable"))
    backup = FailingEvidenceModel(ConnectionError("backup unavailable"))
    gateway = ModelGateway([
        ModelProvider("primary", primary), ModelProvider("backup", backup),
    ], total_timeout=2)

    state, reply = ask_with_evidence_gateway("两次采血最少间隔多久", [hit], gateway)

    assert len(primary.calls) == len(backup.calls) == 1
    assert state["result"]["status"] == "evidence_check_failed"
    assert state["result"]["retrieval_hit_count"] == 1
    assert state["result"]["citations"] == []
    assert state["retrieved_context"] == reply.facts["retrieved_context"] == []
    assert hit["text"] not in state["messages"][-1].content


def test_invalid_primary_evidence_contract_does_not_switch_to_backup():
    hit = {"doc_id": "law", "text": "两次采血间隔不少于六个月。",
           "source": "https://example.org/law"}
    primary = EvidenceModel('{"supported":true,"supporting_doc_ids":["invented"]}')
    backup = EvidenceModel('{"supported":true,"supporting_doc_ids":["law"]}')
    gateway = ModelGateway([
        ModelProvider("primary", primary), ModelProvider("backup", backup),
    ], total_timeout=2)

    state, reply = ask_with_evidence_gateway("两次采血最少间隔多久", [hit], gateway)

    assert len(primary.calls) == 1
    assert backup.calls == []
    assert state["result"]["status"] == "evidence_check_failed"
    assert state["retrieved_context"] == reply.facts["retrieved_context"] == []


def test_unsupported_personal_fact_drops_related_process_snippet_before_reply():
    hit = {"doc_id": "points-process", "text": "个人积分余额应从业务接口查询。",
           "source": "demo://blood-station/points-delivery", "source_kind": "synthetic"}
    state, evidence, reply = ask(
        "我的积分现在是多少", [hit],
        '{"supported":false,"supporting_doc_ids":[]}',
    )
    assert len(evidence.calls) == 1
    assert evidence.calls[0]["question"] == "我的积分现在是多少"
    assert state["result"]["status"] == "no_evidence"
    assert state["result"]["retrieval_hit_count"] == 1
    assert state["result"]["citations"] == []
    assert state["retrieved_context"] == reply.facts["retrieved_context"] == []
    assert state["response_info"]["mode"] == "llm"


def test_supported_question_only_hands_selected_original_snippet_to_reply():
    hits = [
        {"doc_id": "law", "text": "两次采血间隔不少于六个月。",
         "source": "https://example.org/law", "source_kind": "public", "score": 0.7},
        {"doc_id": "care", "text": "献血后适当休息。",
         "source": "https://example.org/care", "source_kind": "public", "score": 0.6},
    ]
    state, _, reply = ask(
        "两次采血最少间隔多久", hits,
        '{"supported":true,"supporting_doc_ids":["law"]}',
    )
    assert state["result"]["status"] == "grounded"
    assert state["result"]["citations"] == ["https://example.org/law"]
    assert reply.facts["retrieved_context"] == [hits[0]]
    assert "https://example.org/care" not in state["messages"][-1].content


@pytest.mark.parametrize("output", [
    '{"supported":true,"supporting_doc_ids":["invented"]}',
    '{"supported":true,"supporting_doc_ids":[]}',
    '{"supported":false,"supporting_doc_ids":["law"]}',
    '{"supported":true,"supporting_doc_ids":["law","law"]}',
    "not-json",
])
def test_invalid_evidence_model_output_fails_closed(output):
    hit = {"doc_id": "law", "text": "两次采血间隔不少于六个月。",
           "source": "https://example.org/law"}
    state, _, reply = ask("两次采血最少间隔多久", [hit], output)
    assert state["result"]["status"] == "evidence_check_failed"
    assert state["retrieved_context"] == reply.facts["retrieved_context"] == []
    assert state["result"]["citations"] == []
    assert state["response_info"]["mode"] == "llm"
    assert "核验暂不可用" in state["messages"][-1].content


def test_no_hits_skips_evidence_model_and_still_reports_no_evidence():
    state, evidence, _ = ask("献血前饮食", [], "not-json")
    assert evidence.calls == []
    assert state["result"]["status"] == "no_evidence"
    assert state["result"]["retrieval_hit_count"] == 0


def test_evidence_verdict_rejects_extra_fields_and_unrecognized_ids():
    with pytest.raises(ValidationError):
        EvidenceAssessor._validate(
            '{"supported":true,"supporting_doc_ids":["law"],"answer":"猜测"}', {"law"},
        )
    with pytest.raises(TypeError, match="未知"):
        EvidenceAssessor._validate(
            '{"supported":true,"supporting_doc_ids":["other"]}', {"law"},
        )


def test_batch_evaluation_keeps_model_error_separate_from_rejection():
    cases = [
        {"id": "P", "group": "static", "query": "采血间隔", "sources": ["https://example.org/law"]},
        {"id": "N", "group": "dynamic", "query": "我的积分余额", "sources": []},
        {"id": "E", "group": "dynamic", "query": "停车费", "sources": []},
    ]

    class BatchRetriever:
        def search(self, _query, top_k):
            assert top_k == 4
            return [{"doc_id": "law", "source": "https://example.org/law",
                     "text": "两次采血间隔不少于六个月"}]

    class BatchAssessor:
        def assess(self, query, hits):
            if query == "停车费":
                raise EvidenceCheckUnavailable({"mode": "unavailable", "error_type": "timeout"})
            return (hits if query == "采血间隔" else []), {"mode": "llm"}

    report = evaluate_evidence(cases, BatchRetriever(), BatchAssessor())
    assert report["metrics"]["positive_source_supported_rate"] == 1.0
    assert report["metrics"]["negative_rejection_rate"] == 0.5
    assert report["metrics"]["check_failed_count"] == 1
    assert [row["passed"] for row in report["cases"]] == [True, True, False]


def test_targeted_evidence_selection_keeps_original_order_and_rejects_unknown_ids():
    cases = [{"id": "P11"}, {"id": "P12"}, {"id": "P13"}]
    assert select_cases(cases, ["P13", "P12"]) == cases[1:]
    assert select_cases(cases, None) is cases
    with pytest.raises(ValueError, match="P99"):
        select_cases(cases, ["P99"])


def test_evidence_report_is_private_and_does_not_overwrite(tmp_path):
    target = tmp_path / "reports" / "one.json"
    save_report({"selected_case_ids": ["P12"]}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"selected_case_ids": ["P12"]}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        save_report({"selected_case_ids": ["P13"]}, target)
