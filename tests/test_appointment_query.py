"""已提交预约的查询属于预约子 Agent 的只读动作，而不是重新开启预约。"""

import json

from blood_agents.conversation import ConversationGraph
from blood_agents.graph import build_graph
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from preview_reply import PreviewReplyModel

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


def setup(router=None):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = ConversationGraph(build_graph(
        repo, router or IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False,
                                    evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver(),
    ))
    return repo, api, graph


def ask(graph, text, *, user="U1001", session="appointment-query"):
    return graph.invoke(
        {"user_id": user, "messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": session}},
    )


def test_query_after_booking_reads_submitted_appointment_without_new_write():
    repo, api, graph = setup()
    day = api.site_slots[0]["slots"][0]["date"]
    preview = ask(graph, f"预约 {day} 下午到市中心献血屋")
    assert preview["result"]["status"] == "awaiting_confirmation"
    booked = ask(graph, "确认预约")
    appointment_id = booked["result"]["data"]["appointment_id"]

    queried = ask(graph, "帮我查询一下我的预约信息")
    assert queried["intent"] == "appointment"
    assert queried["understanding"]["action"] == "query"
    assert queried["result"]["status"] == "appointments_found"
    assert queried["result"]["appointments"][0]["appointment_id"] == appointment_id
    assert queried["response_info"]["mode"] == "unavailable"
    assert appointment_id not in queried["messages"][-1].content
    assert [item["tool"] for item in queried["tool_trace"]] == ["query_user_appointments"]
    assert queried["active_agent"] is None
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 1

    # 当前预约存在 Mock SQLite 中，不依赖刚才的聊天记录或 session checkpoint。
    another_session = ask(graph, "查询我的预约信息", session="another-session")
    assert another_session["result"]["appointments"][0]["appointment_id"] == appointment_id


def test_query_without_booking_and_other_user_do_not_leak_appointments():
    _, api, graph = setup()
    empty = ask(graph, "查询我的预约信息")
    assert empty["result"]["status"] == "no_appointments"
    assert empty["response_info"]["mode"] == "unavailable"
    assert [item["tool"] for item in empty["tool_trace"]] == ["query_user_appointments"]

    day = api.site_slots[0]["slots"][0]["date"]
    ask(graph, f"预约 {day} 上午到市中心献血屋")
    ask(graph, "确认预约")
    other = ask(graph, "查询我的预约信息", user="OTHER")
    assert other["result"]["status"] == "query_failed"
    assert other["result"]["code"] == "USER_NOT_FOUND"
    assert other["response_info"]["mode"] == "unavailable"
    assert "预约号" not in other["messages"][-1].content


def test_read_only_query_preserves_unfinished_booking_preview():
    repo, api, graph = setup()
    day = api.site_slots[0]["slots"][0]["date"]
    preview = ask(graph, f"预约 {day} 上午到市中心献血屋")
    queried = ask(graph, "查一下我的预约记录")
    assert queried["result"]["status"] == "no_appointments"
    assert queried["active_agent"] == "appointment"
    assert queried["appointment_state"] == preview["appointment_state"]
    assert queried["agent_handoff"]["control"] == "user"
    assert [item["tool"] for item in queried["tool_trace"]] == ["query_user_appointments"]
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
    confirmed = ask(graph, "确认预约")
    assert confirmed["result"]["success"] is True


def test_explicit_appointment_query_overrides_model_misrouting_and_collect_action():
    class WrongDecisionModel:
        def __init__(self):
            self.replies = iter([{"intent": "records"}, {"action": "collect"}])

        def invoke(self, _messages):
            return AIMessage(content=json.dumps(next(self.replies), ensure_ascii=False))

    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    repo.create_appointment("U1001", "市中心献血屋", api.site_slots[0]["slots"][0]["date"],
                            "上午", "seed-query")
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(WrongDecisionModel()), legacy_api=api, checkpointer=MemorySaver(),
    ))
    result = ask(graph, "查询我的预约记录")
    assert result["intent"] == "appointment"
    assert result["understanding"]["action"] == "query"
    assert result["model_info"]["navigation"]["guardrail"] == "explicit_appointment_query"
    assert result["model_info"]["appointment"]["action_guardrail"] == "explicit_appointment_query"
    assert [item["tool"] for item in result["tool_trace"]] == ["query_user_appointments"]


def test_availability_question_is_not_a_submitted_appointment_query():
    router = IntentRouter()
    assert router.route("查询我的预约信息") == "appointment"
    assert router.interpret_appointment("查询我的预约信息")[0].action == "query"
    assert router.interpret_appointment("查明天下午能否预约")[0].action == "collect"
