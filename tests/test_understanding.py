"""用可控模型响应验证语义理解和编排；不把模拟输出视为真实模型准确率。"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from blood_agents.graph import build_graph
from blood_agents.model_gateway import ModelGateway, ModelProvider
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from preview_reply import PreviewReplyModel

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=json.dumps(reply, ensure_ascii=False))


def setup(replies):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    model = ScriptedModel(replies)
    graph = build_graph(
        repo, IntentRouter(model, response_llm=PreviewReplyModel(), risk_llm=False,
                           evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver()
    )
    return repo, api, model, graph


def ask(graph, text):
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": "semantic-test"}},
    )


def test_model_drives_natural_language_booking_and_context():
    repo, api, model, graph = setup([])
    day = api.site_slots[0]["slots"][0]["date"]
    model.replies = iter([
        {"intent": "appointment"},
        {},
        {"intent": "appointment"},
        {"slots": {"date": day, "site": "市中心献血屋", "time_slot": "下午"}},
        {"intent": "appointment"},
        {"action": "confirm"},
    ])
    ask(graph, "给我安排一次捐血吧")
    preview = ask(graph, "沿用之前的地方，明儿吃过午饭再去")
    assert preview["result"]["status"] == "awaiting_confirmation"
    assert preview["model_info"]["mode"] == "llm"
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
    result = ask(graph, "就按刚才展示的安排提交吧")
    assert result["result"]["success"] is True
    assert len(model.requests) == 6
    second_context = json.loads(model.requests[3][1].content)["context"]
    assert second_context["current_slots"]["preferred_site"] == "市中心献血屋"
    assert second_context["availability"]


@pytest.mark.parametrize("reply", [
    {"intent": "delete_user"},
    {"slots": {"date": "2026-02-30"}},
    {"slots": {"user_id": "U2000"}},
    {"action": "submit_now"},
    TimeoutError("simulated timeout"),
])
def test_invalid_model_output_or_timeout_falls_back(reply):
    parsed, info = IntentRouter(ScriptedModel([reply])).interpret_appointment("我要预约明天上午")
    assert parsed.action == "collect"
    assert parsed.slots.time_slot == "上午"
    assert info["mode"] == "rules_fallback"
    assert info["error_type"]


@pytest.mark.parametrize("use_gateway", [False, True])
def test_complete_fenced_json_is_accepted_without_allowing_extra_text(use_gateway):
    class RawModel:
        def __init__(self, content):
            self.content = content

        def invoke(self, _messages):
            return AIMessage(content=self.content)

    fenced = '```json\n{"slots": {"time_slot": "下午"}, "action": "collect"}\n```'
    llm = RawModel(fenced)
    if use_gateway:
        llm = ModelGateway([ModelProvider("primary", llm)])
    parsed, info = IntentRouter(llm).interpret_appointment("下午")
    assert info["mode"] == "llm"
    assert parsed.slots.time_slot == "下午"

    invalid = RawModel("请看：\n" + fenced)
    if use_gateway:
        invalid = ModelGateway([ModelProvider("primary", invalid)])
    _, invalid_info = IntentRouter(invalid).interpret_appointment("下午")
    assert invalid_info["mode"] == "rules_fallback"


def test_changed_slots_and_confirm_requires_a_new_preview():
    repo, api, model, graph = setup([])
    day = api.site_slots[0]["slots"][0]["date"]
    model.replies = iter([
        {"intent": "appointment"},
        {"slots": {"date": day, "site": "市中心献血屋", "time_slot": "上午"}},
        {"intent": "appointment"},
        {"slots": {"time_slot": "下午"}, "action": "confirm"},
        {"intent": "appointment"},
        {"action": "confirm"},
    ])
    ask(graph, "预约")
    changed = ask(graph, "改到下午并提交")
    assert changed["result"]["status"] == "awaiting_confirmation"
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
    assert ask(graph, "确认预约")["result"]["data"]["time_slot"] == "下午"


def test_rules_do_not_treat_ambiguous_acknowledgement_as_submission():
    parsed, _ = IntentRouter().interpret_appointment("好的", slots={"status": "awaiting_confirmation"})
    assert parsed.action == "collect"
    assert IntentRouter().interpret_appointment("确认预约")[0].action == "confirm"


def test_navigation_prompt_and_greeting_intent_are_distinct_from_faq():
    model = ScriptedModel([{"intent": "general"}])
    parsed, info = IntentRouter(model).navigate("你好")
    assert parsed.intent == "general" and info["mode"] == "llm"
    system_prompt = model.requests[0][0].content
    assert "主导航智能体" in system_prompt
    assert "不要把问候归入 faq" in system_prompt
    assert '"general"' in system_prompt
    assert '"slots"' not in system_prompt and '"action"' not in system_prompt
    assert IntentRouter().route("你好") == "general"
    assert IntentRouter().route("我想去献血") == "appointment"
    assert IntentRouter().route("献血前可以空腹吗") == "faq"


def test_graph_greeting_uses_only_navigation_model_and_no_faq_tools():
    _, _, model, graph = setup([{"intent": "general"}])
    state = ask(graph, "你好")
    assert state["response_info"]["mode"] == "unavailable"
    assert state["intent"] == "general"
    assert state["tool_trace"] == []
    assert len(model.requests) == 1


def test_explicit_greeting_guard_rejects_model_faq_misroute():
    _, _, _, graph = setup([{"intent": "faq"}])
    state = ask(graph, "你好")
    assert state["intent"] == "general"
    assert state["model_info"]["guardrail"] == "explicit_general_turn"
    assert state["tool_trace"] == []


def test_appointment_prompt_is_owned_by_child_semantic_node():
    model = ScriptedModel([{}])
    parsed, info = IntentRouter(model).interpret_appointment("明天下午")
    assert parsed.action == "collect" and info["mode"] == "llm"
    system_prompt = model.requests[0][0].content
    assert "预约子智能体" in system_prompt
    assert '"slots"' in system_prompt and '"action"' in system_prompt
    assert '"intent"' not in system_prompt


def test_explicit_relative_dates_survive_incomplete_model_output_across_booking_turns():
    """模型漏槽位时，用户明确说出的日期仍要覆盖旧状态，且不能沿用旧时段。"""
    repo, api, model, graph = setup([])
    tomorrow = api.site_slots[0]["slots"][0]["date"]
    day_after = api.site_slots[1]["slots"][0]["date"]
    model.replies = iter([
        {"intent": "appointment"}, {},
        {"intent": "appointment"}, {"slots": {"site": "大学城献血车", "time_slot": "下午"}},
        {"intent": "appointment"}, {"slots": {"date": tomorrow, "site": "大学城献血车"}},
        {"intent": "appointment"}, {"slots": {"time_slot": "上午"}},
    ])

    first = ask(graph, "我想明天献血")
    assert first["slots"]["date"] == tomorrow
    assert "日期" not in "、".join(first["result"]["missing"])
    assert first["model_info"]["appointment"]["corrected_slots"] == ["date"]

    unavailable = ask(graph, "大学城献血车 明天下午的")
    assert unavailable["result"]["status"] == "slot_unavailable"
    assert unavailable["slots"]["time_slot"] == "下午"

    changed_day = ask(graph, "那大学城献血车后天可以约吗？")
    assert changed_day["slots"]["date"] == day_after
    assert "date" in changed_day["model_info"]["appointment"]["corrected_slots"]
    assert "time_slot" not in changed_day["slots"]
    assert changed_day["result"]["status"] == "need_more_info"
    assert changed_day["result"]["missing"] == ["时间段（上午/下午或具体时间）"]
    assert changed_day["response_info"]["mode"] == "unavailable"
    assert changed_day["result"]["availability"][1]["slots"][0]["date"] == day_after

    preview = ask(graph, "上午")
    assert preview["result"]["status"] == "awaiting_confirmation"
    assert preview["result"]["payload_preview"]["date"] == day_after
    assert preview["result"]["payload_preview"]["time_slot"] == "上午"
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0


def test_offline_booking_recognizes_bare_site_and_relative_day():
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = build_graph(repo, IntentRouter(), legacy_api=api, checkpointer=MemorySaver())

    assert ask(graph, "我想明天献血")["slots"]["date"] == api.site_slots[0]["slots"][0]["date"]
    selected = ask(graph, "大学城献血车 明天下午的")
    assert selected["slots"]["site"] == "大学城献血车"
    assert selected["result"]["status"] == "slot_unavailable"
    changed = ask(graph, "那大学城献血车后天可以约吗？")
    assert changed["slots"]["date"] == str(datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=2))
    assert changed["result"]["status"] == "need_more_info"


def test_first_booking_reply_echoes_recognized_date_and_filters_other_days():
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = build_graph(repo, IntentRouter(), legacy_api=api, checkpointer=MemorySaver())

    state = ask(graph, "我想明天来献血")
    tomorrow = api.site_slots[0]["slots"][0]["date"]
    assert state["intent"] == "appointment"
    assert state["slots"]["date"] == tomorrow
    assert state["result"]["missing"] == [
        "时间段（上午/下午或具体时间）", "献血点（如 市中心献血屋）",
    ]
    assert state["response_info"]["mode"] == "unavailable"
    assert state["result"]["availability"][0]["slots"][0]["date"] == tomorrow


def test_ambiguous_relative_date_and_period_are_not_selected_by_rules():
    parsed, _ = IntentRouter().interpret_appointment("明天上午还是后天下午可以约？")
    assert parsed.slots.date is None
    assert parsed.slots.time_slot is None


def test_abandon_tomorrow_and_ask_day_after_uses_today_not_previous_turn():
    """改问后天时，模型的错误日期及旧方案的时段/网点都不能被沿用。"""
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    model = ScriptedModel([{"slots": {
        "date": str(today + timedelta(days=3)),
        "time_slot": "下午", "site": "市中心献血屋",
    }, "action": "cancel"}])
    parsed, info = IntentRouter(model).interpret_appointment(
        "我没办法去市中心献血屋，那我不预约明天下午了，后天有可以预约的吗",
        availability=[{"site": "市中心献血屋", "slots": []},
                      {"site": "大学城献血车", "slots": []}],
    )
    assert parsed.slots.date == str(today + timedelta(days=2))
    assert parsed.slots.time_slot is None
    assert parsed.slots.site is None
    assert parsed.action == "collect"
    assert info["slot_guardrail"] == "explicit_user_slots"
    assert info["action_guardrail"] == "relative_date_change"


def test_booking_followup_abandons_tomorrow_and_lists_day_after_options():
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = build_graph(repo, IntentRouter(), legacy_api=api, checkpointer=MemorySaver())
    first = ask(graph, "我想预约明天下午的献血")
    assert first["slots"]["date"] == api.site_slots[0]["slots"][0]["date"]
    followup = ask(graph, "我没办法去市中心献血屋，那我不预约明天下午了，后天有可以预约的吗")
    assert followup["slots"]["date"] == api.site_slots[1]["slots"][0]["date"]
    assert "time_slot" not in followup["slots"]
    assert "site" not in followup["slots"]
    assert followup["result"]["status"] == "need_more_info"


@pytest.mark.parametrize("model_site", [None, "市中心献血屋"])
def test_rejected_site_is_not_corrected_back_as_selected_site(model_site):
    """模型输出 site=null 时，提到“去不了”的网点不能被字面匹配强行补回。"""
    tomorrow = str(datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1))
    model = ScriptedModel([{"slots": {"date": tomorrow, "site": model_site,
                                      "time_slot": "下午"}, "action": "collect"}])
    parsed, info = IntentRouter(model).interpret_appointment(
        "还有其他网点吗？明天去不了市中心献血屋",
        availability=[{"site": "市中心献血屋", "slots": []},
                      {"site": "大学城献血车", "slots": []}],
    )
    assert parsed.slots.site is None
    assert parsed.slots.date == tomorrow
    assert info["rejected_sites"] == ["市中心献血屋"]


def test_langchain_client_serialization_without_external_api():
    """通过 HTTP Mock 验证实际 ChatOpenAI 请求/响应处理，不发出网络请求。"""
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "mock-completion", "object": "chat.completion", "created": 1,
            "model": "demo-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"intent":"points"}',
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        llm = ChatOpenAI(
            model="demo-model", api_key="test-only", base_url="http://mock.local/v1",
            http_client=client, max_retries=0, timeout=1,
        )
        parsed, info = IntentRouter(llm).navigate("我有多少分")
    assert parsed.intent == "points"
    assert info["mode"] == "llm"
    assert requests[0]["messages"][0]["role"] == "system"
