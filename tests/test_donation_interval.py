"""个人下次献血时间：RAG 规则、登录记录、模型规划与日期计算协作。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.knowledge.donation_interval import (
    INTERVAL_RULE_SOURCE,
    calculate_next_donation_dates,
    is_personal_next_donation_question,
)
from blood_agents.knowledge.rag import default_retriever
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


class Navigation:
    def __init__(self, intent="faq"):
        self.intent = intent

    def invoke(self, messages):
        return AIMessage(content=json.dumps({"intent": self.intent}))


class IntervalEvidence:
    """只放行实际召回的间隔片段，模拟模型证据核验的结构化结果。"""

    def __init__(self):
        self.request = None

    def invoke(self, messages):
        self.request = json.loads(messages[-1].content)
        ids = [item["doc_id"] for item in self.request["passages"]
               if item["source"] == INTERVAL_RULE_SOURCE]
        return AIMessage(content=json.dumps({
            "supported": bool(ids), "supporting_doc_ids": ids,
        }))


class IntervalPlan:
    def __init__(self, *, needs_record=True, target="both"):
        self.needs_record = needs_record
        self.target = target
        self.request = None

    def invoke(self, messages):
        self.request = json.loads(messages[-1].content)
        return AIMessage(content=json.dumps({
            "needs_personal_record": self.needs_record,
            "target_donation_type": self.target,
        }, ensure_ascii=False))


class IntervalReply:
    def __init__(self):
        self.facts = None

    def invoke(self, messages):
        self.facts = json.loads(messages[-1].content)
        result = self.facts["result"]
        personal = result.get("personal_interval") or {}
        source = INTERVAL_RULE_SOURCE
        title = next(hit["title"] for hit in self.facts["retrieved_context"]
                     if hit["source"] == source)
        if personal.get("status") == "calculated":
            dates = {item["donation_type"]: item["earliest_date"]
                     for item in personal["earliest_dates"]}
            if len(dates) == 2:
                answer = (
                    f"您上次于 {personal['last_donation_date']} 捐献{personal['last_donation_type']}。"
                    f"按最短间隔参考：全血 {dates['全血']}，机采 {dates['机采']}。"
                )
            else:
                donation_type, available_date = next(iter(dates.items()))
                answer = (f"您上次于 {personal['last_donation_date']} 捐献{personal['last_donation_type']}，"
                          f"{donation_type}最早参考日期为 {available_date}。")
            return AIMessage(content=f"{answer}最终以血站健康评估为准。依据《{title}》：{source}")
        if result["status"] == "personal_interval_no_records":
            answer = "当前登录用户本轮没有查到献血记录，因此无法计算个人下次日期。"
            return AIMessage(content=f"{answer}一般规则见《{title}》：{source}")
        return AIMessage(content=f"献血间隔规划暂不可用。来源：{source}")


def setup(*, user="U1001", plan=None, navigation=None, evidence=False):
    repo = BloodRepository()
    repo.seed_demo_data()
    planner = plan or IntervalPlan()
    reply = IntervalReply()
    router = IntentRouter(
        llm=navigation or Navigation(), response_llm=reply, tool_llm=planner,
        risk_llm=False, evidence_llm=evidence,
    )
    graph = ConversationGraph(build_graph(
        repo, router, retriever=default_retriever(), checkpointer=MemorySaver(),
    ))
    return repo, graph, planner, reply, user


def ask(graph, text, user="U1001"):
    return graph.invoke(
        {"user_id": user, "messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": f"interval-{user}-{text}"}},
    )


@pytest.mark.parametrize(("last_type", "target", "expected"), [
    ("全血", "全血", "2026-06-13"),
    ("全血", "机采", "2026-03-15"),
    ("机采", "机采", "2025-12-29"),
    ("机采", "全血", "2026-01-12"),
])
def test_interval_calculation_uses_2026_rule_day_counts(
    last_type, target, expected,
):
    result = calculate_next_donation_dates("2025-12-15", last_type, target)
    assert result["earliest_dates"][0]["earliest_date"] == expected


def test_2026_rule_uses_180_days_not_six_calendar_months():
    result = calculate_next_donation_dates("2025-08-31", "全血", "全血")
    assert result["earliest_dates"][0]["earliest_date"] == "2026-02-27"
    assert result["earliest_dates"][0]["interval_amount"] == 180


def test_personal_next_donation_combines_rag_plan_and_current_user_record():
    _repo, graph, planner, reply, _user = setup()
    state = ask(graph, "我想查询下次献血时间")
    assert state["intent"] == "faq"
    assert state["result"]["status"] == "personal_interval_calculated"
    assert state["result"]["personal_interval"]["last_donation_date"] == "2025-12-15"
    assert state["result"]["personal_interval"]["last_donation_type"] == "全血"
    assert state["result"]["personal_interval"]["earliest_dates"] == [
        {"donation_type": "全血", "earliest_date": "2026-06-13",
         "interval_unit": "days", "interval_amount": 180},
        {"donation_type": "机采", "earliest_date": "2026-03-15",
         "interval_unit": "days", "interval_amount": 90},
    ]
    assert planner.request["user_message"] == "我想查询下次献血时间"
    assert planner.request["verified_interval_evidence"]
    assert planner.request["verified_interval_evidence"][0]["source"] == INTERVAL_RULE_SOURCE
    assert [item["tool"] for item in state["tool_trace"]] == ["knowledge_search", "query_records"]
    assert "2026-06-13" in state["messages"][-1].content
    assert "2026-03-15" in state["messages"][-1].content
    assert INTERVAL_RULE_SOURCE in state["messages"][-1].content
    assert reply.facts["retrieved_context"]


@pytest.mark.parametrize("navigation_intent", ["records", "general"])
def test_exact_user_question_routes_to_rag_even_when_navigation_model_misroutes(navigation_intent):
    question = "帮我查询一下我什么时候可以献下一次血"
    evidence = IntervalEvidence()
    _repo, graph, planner, reply, _user = setup(
        navigation=Navigation(navigation_intent), evidence=evidence,
    )
    state = ask(graph, question)

    assert state["intent"] == "faq"
    assert state["model_info"]["navigation"]["guardrail"] == (
        "personal_next_donation_requires_interval_and_records"
    )
    assert state["result"]["status"] == "personal_interval_calculated"
    assert [item["tool"] for item in state["tool_trace"]] == [
        "knowledge_search", "query_records",
    ]
    assert evidence.request["question"] == question
    assert planner.request["verified_interval_evidence"][0]["source"] == INTERVAL_RULE_SOURCE
    assert reply.facts["result"]["personal_interval"]["last_donation_date"] == "2025-12-15"
    assert "2026-06-13" in state["messages"][-1].content


def test_offline_navigation_uses_same_cross_domain_guardrail_without_model():
    router = IntentRouter()
    assert router.route("帮我查询一下我什么时候可以献下一次血") == "faq"
    assert router.route("帮我查询我的献血记录") == "records"
    assert router.route("帮我查询下次献血预约信息") == "appointment"
    assert not is_personal_next_donation_question("我什么时候献过血？")


def test_clear_personal_next_date_still_queries_records_if_planner_says_general_rule():
    _repo, graph, planner, _reply, _user = setup(plan=IntervalPlan(needs_record=False))
    state = ask(graph, "帮我查询一下我什么时候可以献下一次血")
    assert planner.request["verified_interval_evidence"]
    assert state["result"]["status"] == "personal_interval_calculated"
    assert [item["tool"] for item in state["tool_trace"]] == [
        "knowledge_search", "query_records",
    ]


def test_explicit_target_type_limits_calculation_even_if_planner_says_both():
    _repo, graph, _planner, _reply, _user = setup(plan=IntervalPlan(target="both"))
    state = ask(graph, "我下次想献全血，最早什么时候？")
    personal = state["result"]["personal_interval"]
    assert personal["target_donation_type"] == "全血"
    assert [item["donation_type"] for item in personal["earliest_dates"]] == ["全血"]


def test_user_without_records_gets_no_personal_date_and_record_tool_is_still_traced():
    _repo, graph, _planner, _reply, _user = setup(user="U1002")
    state = ask(graph, "我下次什么时候能献血？", user="U1002")
    assert state["result"]["status"] == "personal_interval_no_records"
    assert [item["tool"] for item in state["tool_trace"]] == ["knowledge_search", "query_records"]
    assert "没有查到献血记录" in state["messages"][-1].content
    assert "2026-" not in state["messages"][-1].content


def test_unknown_latest_record_type_does_not_fall_back_to_older_record():
    records = [
        {"donated_at": "2025-12-15", "donation_type": "全血"},
        {"donated_at": "2026-01-01", "donation_type": None},
    ]
    latest = max(records, key=lambda item: item["donated_at"])
    assert calculate_next_donation_dates(latest["donated_at"], latest["donation_type"], "both") is None
