"""验证预约模型提出只读工具请求，以及回复发布前的风控闭环。"""

import json

import pytest
from blood_agents.conversation import ConversationGraph
from blood_agents.graph import build_graph
from blood_agents.reply_composer import UNAVAILABLE_MESSAGE
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


@pytest.fixture(autouse=True)
def _enable_legacy_risk_tests(monkeypatch):
    """内容风控按配置启用；不让额外模型调用影响其他测试。"""
    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "model")


def _graph(**models):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    router = IntentRouter(**models)
    return ConversationGraph(build_graph(repo, router, legacy_api=api, checkpointer=MemorySaver()))


def _ask(graph, question, session):
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": session}},
    )


class FactsReply:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        facts = json.loads(messages[-1].content)
        return AIMessage(content="我查到：" + "、".join(facts["required_facts"]))


def test_appointment_model_requests_read_tools_without_identity_or_write_access():
    class Planner:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content='{"tools":["query_sites","query_appointment_history"]}')

    planner = Planner()
    graph = _graph(tool_llm=planner)
    state = _ask(graph, "我要预约，看看目前有哪些网点", "planned-reads")
    assert state["tool_plan_info"]["mode"] == "llm"
    assert state["tool_plan_info"]["requested_tools"] == ["query_sites", "query_appointment_history"]
    assert [item["tool"] for item in state["tool_trace"]].count("query_sites") == 1
    assert [item["tool"] for item in state["tool_trace"]].count("query_appointment_history") == 1
    assert "U1001" not in planner.messages[-1].content
    assert "submit_appointment" not in state["tool_plan_info"]["requested_tools"]


def test_tool_planner_rejects_write_request_and_does_not_execute_it():
    class BadPlanner:
        def invoke(self, _messages):
            return AIMessage(content='{"tools":["submit_appointment"]}')

    state = _ask(_graph(tool_llm=BadPlanner()), "我要预约", "bad-plan")
    assert state["tool_plan_info"]["mode"] == "rules_fallback"
    assert state["tool_plan_info"]["requested_tools"] == []
    assert all(item["tool"] != "submit_appointment" for item in state["tool_trace"])


def test_risk_agent_blocks_first_draft_and_releases_only_reviewed_revision():
    class Risk:
        def __init__(self):
            self.calls = 0
            self.messages = []

        def invoke(self, messages):
            self.calls += 1
            self.messages.append(messages)
            return AIMessage(content=json.dumps({"allow": self.calls == 2,
                                                 "reason": "第一稿需要更谨慎"}))

    reply, risk = FactsReply(), Risk()
    state = _ask(_graph(response_llm=reply, risk_llm=risk), "查询我的积分", "risk-rewrite")
    assert reply.calls == 2 and risk.calls == 2
    assert state["response_info"]["mode"] == "llm"
    assert state["response_info"]["risk"]["allow"] is True
    assert "1200" in state["messages"][-1].content
    assert len(state["messages"]) == 2  # 第一稿从未发布到会话。
    assert "违法有害内容" in risk.messages[0][0].content
    assert "不要复核业务字段是否逐字出现" in risk.messages[0][0].content
    assert set(json.loads(risk.messages[0][-1].content)) == {"user_message", "intent", "candidate"}


def test_second_risk_block_uses_model_refusal_and_never_replays_tools():
    class Risk:
        def invoke(self, messages):
            if "候选回复未通过内容安全审查" in messages[0].content:
                return AIMessage(content="这次我没法可靠地回答，请稍后重试或咨询血站人员。")
            if "这次我没法可靠地回答" in messages[-1].content:
                return AIMessage(content='{"allow":true,"reason":""}')
            return AIMessage(content='{"allow":false,"reason":"信息需要人工核实"}')

    reply = FactsReply()
    state = _ask(_graph(response_llm=reply, risk_llm=Risk()), "查询我的积分", "risk-refusal")
    assert reply.calls == 2
    assert state["response_info"]["mode"] == "risk_refusal"
    assert "1200" not in state["messages"][-1].content
    assert "可靠" in state["messages"][-1].content
    assert [item["tool"] for item in state["tool_trace"]] == ["query_points", "list_rewards"]


def test_content_refusal_is_reviewed_before_release():
    class Risk:
        def invoke(self, messages):
            if "候选回复未通过内容安全审查" in messages[0].content:
                return AIMessage(content="这次我不能继续回答。")
            return AIMessage(content='{"allow":false,"reason":"仍不安全"}')

    state = _ask(_graph(response_llm=FactsReply(), risk_llm=Risk()),
                 "查询我的积分", "risk-refusal-rejected")
    assert state["response_info"]["mode"] == "unavailable"
    assert state["response_info"]["error_type"] == "risk_refusal_rejected"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE


def test_unavailable_risk_model_fails_closed():
    class BrokenRisk:
        def invoke(self, _messages):
            raise TimeoutError("risk provider unavailable")

    state = _ask(_graph(response_llm=FactsReply(), risk_llm=BrokenRisk()),
                 "查询我的积分", "risk-unavailable")
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    assert state["response_info"]["mode"] == "unavailable"


def test_custom_reply_model_cannot_silently_skip_risk_agent():
    state = _ask(_graph(response_llm=FactsReply()), "查询我的积分", "risk-default")
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE


def test_risk_rewrite_after_booking_does_not_replay_submit_tool():
    class BookingReply:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            prefix = "预约成功：" if facts["result_success"] is True else "请确认预约："
            return AIMessage(content=prefix + "、".join(facts["required_facts"]))

    class Risk:
        def __init__(self):
            self.success_reviews = 0

        def invoke(self, messages):
            candidate = json.loads(messages[-1].content)["candidate"]
            if "预约成功" in candidate:
                self.success_reviews += 1
                return AIMessage(content=json.dumps({"allow": self.success_reviews == 2,
                                                     "reason": "请复核预约号"}))
            return AIMessage(content='{"allow":true,"reason":""}')

    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    risk = Risk()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=BookingReply(), risk_llm=risk),
        legacy_api=api, checkpointer=MemorySaver(),
    ))
    day = api.site_slots[0]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到市中心献血屋", "risk-submit-once")
    booked = _ask(graph, "确认预约", "risk-submit-once")
    assert booked["result"]["success"] is True
    assert risk.success_reviews == 2
    assert booked["response_info"]["mode"] == "llm"
    assert [item["tool"] for item in booked["tool_trace"]].count("submit_appointment") == 1
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 1


def test_repeated_fact_validation_failure_stays_outside_content_risk_model():
    class InvalidReply:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return AIMessage(content="您有 999 积分，已经兑换成功。")

    class RefusalRisk:
        def invoke(self, messages):
            raise AssertionError("事实校验失败不应调用内容风控 Agent")

    reply = InvalidReply()
    state = _ask(_graph(response_llm=reply, risk_llm=RefusalRisk()),
                 "查询我的积分", "fact-refusal")
    assert reply.calls == 2
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    assert "999" not in state["messages"][-1].content
