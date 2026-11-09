"""验证入口隔离、工具轨迹及预约提交边界；所有接口均为本地 Mock。"""

import json
from threading import Event, Lock, Thread

from blood_agents.conversation import ConversationGraph
from blood_agents.graph import build_graph
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from preview_reply import PreviewReplyModel

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


def setup(api_type=MockLegacyBloodAPI):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = api_type(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver()))
    day = api.site_slots[0]["slots"][0]["date"]
    return graph, api, day


def ask(graph, text, user="U1001", session="shared"):
    return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": user},
                        config={"configurable": {"thread_id": session}})


def test_user_and_session_isolation():
    graph, _, day = setup()
    preview = ask(graph, f"预约 {day} 上午到市中心献血屋")
    assert preview["appointment_state"] == preview["slots"]
    assert preview["slots"]["status"] == "awaiting_confirmation"
    other_user = ask(graph, "查询积分", user="OTHER")
    other_session = ask(graph, "查询积分", session="another")
    assert not other_user.get("slots")
    assert not other_session.get("slots")
    assert len(other_user["messages"]) == 2
    assert len(other_session["messages"]) == 2
    assert ask(graph, "确认预约")["result"]["success"] is True


def test_trace_per_turn_redacted_and_interrupted_booking_resumes():
    graph, _, day = setup()
    preview = ask(graph, f"预约 {day} 上午到市中心献血屋")
    assert preview["active_agent"] == "appointment"
    assert preview["agent_status"] == "waiting"
    assert preview["agent_handoff"]["control"] == "user"
    assert [x["node"] for x in preview["node_trace"]] == [
        "supervisor", "appointment_subgraph", "appointment_understand", "appointment_tool_plan",
        "appointment_agent",
        "appointment_reply", "appointment_handoff",
    ]
    assert [x["tool"] for x in preview["tool_trace"]] == [
        "query_user_profile", "query_sites", "query_appointment_history",
    ]
    points = ask(graph, "查询我的积分")
    assert points["intent"] == "points"
    assert points["slots"] == preview["slots"]
    assert points["appointment_state"] == preview["appointment_state"]
    assert points["active_agent"] == "appointment"
    assert points["agent_handoff"]["agent"] == "points"
    assert points["agent_handoff"]["control"] == "main"
    assert points["node_trace"][-1]["node"] == "main_agent_return"
    assert [x["tool"] for x in points["tool_trace"]] == ["query_points", "list_rewards"]
    success = ask(graph, "确认预约")
    assert success["active_agent"] is None
    assert success["appointment_state"] == {}
    assert success["agent_status"] == "completed"
    assert success["agent_handoff"]["control"] == "main"
    assert success["node_trace"][-1]["node"] == "main_agent_return"
    submits = [x for x in success["tool_trace"] if x["tool"] == "submit_appointment"]
    assert len(submits) == 1 and submits[0]["mutating"]
    assert submits[0]["arguments"]["date"] == day
    trace = json.dumps(success["tool_trace"])
    assert "U1001" not in trace and "mobile" not in trace and "idempotency" not in trace


def test_submission_failure_returned_by_graph_and_traced():
    class RejectAPI(MockLegacyBloodAPI):
        def submit_appointment(self, payload, idempotency_key):
            return {"success": False, "code": "MOCK_REJECTED", "message": "模拟网点暂停预约", "data": {}}

    graph, _, day = setup(RejectAPI)
    ask(graph, f"预约 {day} 上午到市中心献血屋")
    result = ask(graph, "确认预约")
    assert result["response_info"]["mode"] == "unavailable"
    assert "模拟网点暂停预约" not in result["messages"][-1].content
    assert result["result"]["message"] == "模拟网点暂停预约"
    assert result["result"]["code"] == "MOCK_REJECTED"
    assert result["tool_trace"][-1]["status"] == "failed"
    assert result["slots"] == {}


def test_cancel_prevents_confirmation_of_old_preview():
    graph, _, day = setup()
    ask(graph, f"预约 {day} 上午到市中心献血屋")
    cancelled = ask(graph, "退出预约")
    assert cancelled["slots"] == {} and cancelled["tool_trace"] == []
    assert cancelled["result"]["scope"] == "draft"
    later = ask(graph, "确认预约")
    assert later["result"]["status"] == "need_more_info"
    assert all(x["tool"] != "submit_appointment" for x in later["tool_trace"])


def test_cancelling_booked_record_requires_confirmation_and_changes_database():
    graph, api, day = setup()
    ask(graph, f"预约 {day} 下午到市中心献血屋", session="saved-cancel")
    booked = ask(graph, "确认预约", session="saved-cancel")
    appointment_id = booked["result"]["data"]["appointment_id"]

    attempted = ask(graph, f"帮我取消 {day} 下午市中心献血屋的预约", session="saved-cancel")
    assert attempted["result"]["status"] == "awaiting_action_confirmation"
    assert attempted["result"]["appointment"]["appointment_id"] == appointment_id
    assert [item["tool"] for item in attempted["tool_trace"]] == ["query_user_appointments"]
    assert api.get_user_appointments("U1001")["data"]["appointments"][0]["status"] == "booked"
    confirmed = ask(graph, "确认取消", session="saved-cancel")
    assert confirmed["result"]["operation"] == "cancel_appointment"
    assert confirmed["result"]["success"] is True
    assert [item["tool"] for item in confirmed["tool_trace"]] == ["cancel_appointment"]
    assert api.get_user_appointments("U1001")["data"]["appointments"][0]["status"] == "cancelled"


def test_relative_date_cancel_request_previews_without_cancel():
    graph, api, _ = setup()
    slot = api.site_slots[1]["slots"][0]
    created = api.submit_appointment({
        "user_id": "U1001", "site": "大学城献血车", "date": slot["date"],
        "time_slot": slot["time_slot"], "slot_id": slot["slot_id"],
        "donor": {"kind": "self", "name": "小明", "mobile_masked": "138****5678"},
    }, "relative-cancel-test")
    assert created["success"] is True

    state = ask(graph, "帮我取消后天上午大学城献血车的预约", session="relative-cancel")
    assert state["understanding"]["action"] == "cancel"
    assert state["result"]["status"] == "awaiting_action_confirmation"
    assert state["result"]["appointment"]["appointment_id"] == created["data"]["appointment_id"]
    assert api.get_user_appointments("U1001")["data"]["appointments"][0]["status"] == "booked"


def test_explicit_refusal_returns_from_appointment_subgraph_without_submit():
    graph, _, day = setup()
    ask(graph, f"预约 {day} 上午到市中心献血屋")
    refused = ask(graph, "我不想预约了")
    assert refused["active_agent"] is None
    assert refused["agent_handoff"]["status"] == "cancelled"
    assert refused["agent_handoff"]["control"] == "main"
    assert refused["node_trace"][-1]["node"] == "main_agent_return"
    assert all(item["tool"] != "submit_appointment" for item in refused["tool_trace"])


def test_greeting_uses_main_navigation_without_faq_and_preserves_booking(tmp_path):
    repo = BloodRepository(str(tmp_path / "greeting.db"))
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver(),
    ))

    greeting = ask(graph, "你好", session="greeting")
    assert greeting["intent"] == "general"
    assert [item["node"] for item in greeting["node_trace"]] == [
        "supervisor", "main_agent_response", "general_reply",
    ]
    assert greeting["tool_trace"] == []
    assert greeting["response_info"]["mode"] == "unavailable"
    assert greeting["result"]["capabilities"]
    assert greeting["result"]["service_facts"] == {
        "business_transport": "local",
        "business_connection_verified": False,
        "ocr_stage": "deferred_prototype_only",
    }
    assert all("图片" not in capability for capability in greeting["result"]["capabilities"])
    assert "知识库中没有找到" not in greeting["messages"][-1].content

    started = ask(graph, "我要预约", session="greeting")
    during = ask(graph, "你好", session="greeting")
    assert during["intent"] == "general"
    assert during["appointment_state"] == started["appointment_state"]
    assert during["active_agent"] == "appointment"
    assert during["tool_trace"] == []
    assert during["result"]["active_appointment"] is True

    knowledge = ask(graph, "献血前可以空腹吗", session="greeting")
    assert knowledge["intent"] == "faq"
    assert knowledge["appointment_state"] == started["appointment_state"]
    assert knowledge["active_agent"] == "appointment"
    assert knowledge["result"]["citations"]

    resumed = ask(graph, "还是上次网点，选择明天上午", session="greeting")
    assert resumed["intent"] == "appointment"
    assert resumed["result"]["status"] == "awaiting_confirmation"


def test_same_conversation_is_serialized():
    entered, release, second = Event(), Event(), Event()

    class SlowGraph:
        def invoke(self, state, config):
            if state["text"] == "first":
                entered.set()
                release.wait(2)
            else:
                second.set()
            return state

    graph = ConversationGraph(SlowGraph())
    config = {"configurable": {"thread_id": "same"}}
    one = Thread(target=lambda: graph.invoke({"user_id": "U1001", "text": "first"}, config))
    two = Thread(target=lambda: graph.invoke({"user_id": "U1001", "text": "second"}, config))
    one.start()
    try:
        assert entered.wait(1)
        two.start()
        assert not second.wait(0.03)
    finally:
        release.set()
        one.join(2)
        two.join(2)
    assert second.is_set()


def test_distinct_conversations_can_run_in_parallel():
    entered, release = Event(), Event()
    gate = Lock()
    count = {"active": 0, "peak": 0}

    class SlowGraph:
        def invoke(self, state, config):
            with gate:
                count["active"] += 1
                count["peak"] = max(count["peak"], count["active"])
                if count["active"] == 2:
                    entered.set()
            release.wait(2)
            with gate:
                count["active"] -= 1
            return state

    graph = ConversationGraph(SlowGraph())
    workers = [Thread(target=lambda session=session: graph.invoke(
        {"user_id": "U1001", "text": "hello"},
        {"configurable": {"thread_id": session}},
    )) for session in ("session-a", "session-b")]
    for worker in workers:
        worker.start()
    try:
        assert entered.wait(1)
        assert count["peak"] == 2
    finally:
        release.set()
        for worker in workers:
            worker.join(2)


def test_web_callbacks_and_per_load_session(tmp_path):
    from blood_agents.web import ChatApplication, build_ui

    app = ChatApplication(str(tmp_path / "web.db"))
    graph, _, day = setup()
    app.graph = graph
    first = app.respond(f"预约 {day} 上午到市中心献血屋", [], "U1001", "web")
    other = app.respond("查询积分", [], "OTHER", "web")
    assert first[3]["slots"]["status"] == "awaiting_confirmation"
    assert first[3]["trace_id"] and first[3]["session_id"] == "web"
    assert first[3]["user_id"] == "U1001"
    assert first[3]["active_agent"] == "appointment"
    assert first[3]["response_info"]["mode"] == "llm"
    assert first[3]["tool_plan_info"]["requested_tools"] == []
    assert first[3]["agent_handoff"]["control"] == "user"
    assert not other[3]["slots"]
    assert first[3]["tool_trace"]
    hello = app.respond_with_summary("你好", [], "U1001", "web-hello")
    assert "智能回复服务暂时不可用" in hello[1][-1][1]
    assert "知识库中没有找到" not in hello[1][-1][1]
    assert "supervisor → main_agent_response" in hello[4]
    assert "本轮追踪 ID：" in hello[4]
    assert "工具调用：0 次" in hello[4]
    summarized = app.respond_with_summary(f"预约 {day} 上午到市中心献血屋", [], "U1001", "web-summary")
    assert "appointment_subgraph" in summarized[4]
    assert "控制交接：appointment → 等待用户" in summarized[4]
    confirmed = app.respond_with_summary("确认预约", summarized[1], "U1001", "web-summary")
    appointment_id = confirmed[3]["result"]["data"]["appointment_id"]
    queried = app.respond_with_summary("查询我的预约信息", confirmed[1], "U1001", "web-summary")
    assert queried[3]["result"]["appointments"][0]["appointment_id"] == appointment_id
    assert [item["tool"] for item in queried[3]["tool_trace"]] == ["query_user_appointments"]
    assert "query_user_appointments" in queried[4]
    demo = build_ui(app)
    # Gradio/Pydantic 的 JSON Schema 兼容回归：页面启动会读取同一份 API 信息。
    assert demo.get_api_info()
    session_box = next(block for block in demo.blocks.values() if getattr(block, "label", None) == "会话 ID")
    load_fn = session_box.load_event_to_attach[0]
    assert session_box.load_event is not None
    assert load_fn() != load_fn()


def test_api_returns_session_and_uses_header_identity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BLOOD_AGENT_DB", str(tmp_path / "api.db"))
    from blood_agents import api

    graph, _, day = setup()
    monkeypatch.setattr(api, "graph", graph)
    with TestClient(api.app) as client:
        preview = client.post("/chat", json={"message": f"预约 {day} 上午到市中心献血屋"}).json()
        assert preview["session_id"] and preview["trace_id"] and preview["tool_trace"]
        assert preview["response_info"]["mode"] == "llm"
        assert preview["tool_plan_info"]["requested_tools"] == []
        other = client.post("/chat", headers={"X-User-ID": "OTHER"}, json={
            "user_id": "U1001", "session_id": preview["session_id"], "message": "查询积分",
        }).json()
        assert other["intent"] == "points" and other["result"]["success"] is False
        success = client.post("/chat", json={
            "session_id": preview["session_id"], "message": "确认预约",
        }).json()
        assert success["result"]["success"] is True
        new = client.post("/chat", json={"message": "查询积分"}).json()
        assert new["session_id"] != preview["session_id"]


def test_fastapi_entry_serializes_same_session_and_runs_distinct_sessions_in_parallel(
    monkeypatch,
):
    """从真实 /chat ASGI 入口验证会话锁；底层模型用可控闸门替代，不打外部 API。"""
    from fastapi.testclient import TestClient

    from blood_agents import api

    class ProbeGraph:
        def __init__(self):
            self.entered_first = Event()
            self.entered_second = Event()
            self.release = Event()
            self.lock = Lock()
            self.calls = 0
            self.active = 0
            self.peak_active = 0

        def invoke(self, state, config):
            with self.lock:
                self.calls += 1
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                if self.calls == 1:
                    self.entered_first.set()
                elif self.calls == 2:
                    self.entered_second.set()
            try:
                assert self.release.wait(3), "测试闸门未释放"
                return {
                    **state,
                    "messages": [*state["messages"], AIMessage(content="已处理")],
                    "intent": "general",
                    "trace_id": "",
                    "result": {},
                    "model_info": {},
                    "response_info": {},
                    "tool_plan_info": {},
                    "tool_trace": [],
                }
            finally:
                with self.lock:
                    self.active -= 1

    def concurrent_post(session_id, message, responses, errors):
        try:
            # 每个线程使用独立 TestClient，模拟多个并发 HTTP 调用方。
            with TestClient(api.app) as client:
                response = client.post("/chat", json={
                    "user_id": "U1001", "session_id": session_id, "message": message,
                })
                responses.append(response)
        except Exception as exc:  # noqa: BLE001 - 在线程中收集错误，交由 pytest 主线程断言
            errors.append(exc)

    # 两条不同会话应同时进入图执行层。
    parallel_probe = ProbeGraph()
    monkeypatch.setattr(api, "graph", ConversationGraph(parallel_probe))
    parallel_responses, parallel_errors = [], []
    parallel_workers = [
        Thread(target=concurrent_post,
               args=(session, "你好", parallel_responses, parallel_errors))
        for session in ("api-session-a", "api-session-b")
    ]
    for worker in parallel_workers:
        worker.start()
    try:
        assert parallel_probe.entered_second.wait(1)
        assert parallel_probe.peak_active == 2
    finally:
        parallel_probe.release.set()
        for worker in parallel_workers:
            worker.join(3)
    assert not parallel_errors
    assert len(parallel_responses) == 2
    assert all(response.status_code == 200 for response in parallel_responses)
    parallel_trace_ids = {response.json()["trace_id"] for response in parallel_responses}
    assert len(parallel_trace_ids) == 2

    # 同一用户、同一会话的第二个请求必须等第一个请求写完 checkpoint 再进入。
    serial_probe = ProbeGraph()
    monkeypatch.setattr(api, "graph", ConversationGraph(serial_probe))
    serial_responses, serial_errors = [], []
    first = Thread(target=concurrent_post,
                   args=("api-session-same", "第一轮", serial_responses, serial_errors))
    second = Thread(target=concurrent_post,
                    args=("api-session-same", "第二轮", serial_responses, serial_errors))
    first.start()
    try:
        assert serial_probe.entered_first.wait(1)
        second.start()
        # 如果入口没有按用户/会话持锁，第二次 invoke 会在释放前进入。
        assert not serial_probe.entered_second.wait(0.1)
        assert serial_probe.peak_active == 1
    finally:
        serial_probe.release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not serial_errors
    assert len(serial_responses) == 2
    assert all(response.status_code == 200 for response in serial_responses)
    serial_trace_ids = {response.json()["trace_id"] for response in serial_responses}
    assert len(serial_trace_ids) == 2
    assert serial_probe.calls == 2
    assert serial_probe.peak_active == 1
