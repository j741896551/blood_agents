"""短期记忆只投影已有流程状态，不从对话摘要或模型回复猜业务事实。"""

import json

from langchain_core.messages import AIMessage, HumanMessage

from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.orchestration.short_term_memory import project_short_term_memory
from local_business.repository import BloodRepository


def _ask(graph, user_id, session_id, text):
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": session_id}, "metadata": {"entrypoint": "test"}},
    )


def test_short_term_memory_whitelists_flow_fields_and_separates_suspended_task():
    snapshot = project_short_term_memory({
        "active_agent": "points",
        "appointment_state": {
            "status": "awaiting_confirmation", "date": "2026-10-22", "site": "演示献血屋",
            "time_slot": "下午", "donor_name": "合成姓名", "donor_mobile": "13800001234",
            "donor_id_card": "310101199001011234", "idempotency_key": "secret-key",
        },
        "points_state": {
            "kind": "redeem", "reward_id": "R001", "points": 1200,
            "recipient": "合成收货人", "address_detail": "演示地址", "mobile": "13800005678",
            "idempotency_key": "secret-key",
        },
    })
    assert snapshot["source"] == "agent_state"
    assert snapshot["current_task"]["agent"] == "points"
    assert snapshot["current_task"]["collected_params"] == {"reward_id": "R001"}
    assert snapshot["suspended_tasks"][0]["collected_params"] == {
        "date": "2026-10-22", "site": "演示献血屋", "time_slot": "下午",
    }
    encoded = json.dumps(snapshot, ensure_ascii=False)
    for secret in ("合成姓名", "合成收货人", "13800001234", "13800005678",
                   "310101199001011234", "演示地址", "secret-key", "1200"):
        assert secret not in encoded


def test_checkpoint_short_term_memory_reaches_navigation_without_cross_user_leak(tmp_path, monkeypatch):
    class SpyModel:
        def __init__(self):
            self.navigation_contexts = []

        def invoke(self, messages):
            if "主导航智能体" in messages[0].content:
                self.navigation_contexts.append(json.loads(messages[-1].content)["context"])
                return AIMessage(content='{"intent":"points"}')
            return AIMessage(content='{"action":"query"}')

    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "memory-demo.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    first = build_conversational_graph(repo=repo, router=IntentRouter())
    try:
        preview = _ask(first, "U1001", "same-id", "我想预约明天下午献血")
        assert preview["result"]["status"] == "need_more_info"
        assert preview["short_term_memory"]["current_task"]["agent"] == "appointment"
    finally:
        first.close()

    spy = SpyModel()
    reopened = build_conversational_graph(
        repo=BloodRepository(str(path)), router=IntentRouter(llm=spy, risk_llm=False),
    )
    try:
        other = _ask(reopened, "U1002", "same-id", "查询积分")
        assert other["intent"] == "points"
        assert spy.navigation_contexts[-1]["short_term_memory"]["current_task"] is None

        switched = _ask(reopened, "U1001", "same-id", "查询积分")
        assert switched["intent"] == "points"
        memory = spy.navigation_contexts[-1]["short_term_memory"]
        assert memory["active_agent"] == "appointment"
        assert memory["current_task"]["agent"] == "appointment"
        assert memory["current_task"]["collected_params"]["time_slot"] == "下午"
        assert switched["short_term_memory"]["current_task"]["agent"] == "appointment"
    finally:
        reopened.close()


def test_completed_or_exited_task_does_not_remain_in_short_term_memory():
    assert project_short_term_memory({
        "active_agent": None, "appointment_state": {}, "points_state": {},
    })["current_task"] is None
    assert project_short_term_memory({
        "active_agent": None, "appointment_state": {"status": "completed", "site": "旧网点"},
        "points_state": {},
    })["suspended_tasks"] == []


def test_short_term_memory_drops_sensitive_content_in_free_text_slot():
    snapshot = project_short_term_memory({
        "active_agent": "appointment",
        "appointment_state": {"status": "collecting", "date": "2026-10-22",
                              "site": "收货地址：北京市朝阳区幸福路8号"},
    })
    assert snapshot["current_task"]["collected_params"] == {"date": "2026-10-22"}


def test_exiting_unsubmitted_appointment_clears_active_short_term_task():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo=repo, router=IntentRouter())
    try:
        first = _ask(graph, "U1001", "exit-draft", "我想预约明天下午献血")
        assert first["short_term_memory"]["current_task"]["agent"] == "appointment"
        exited = _ask(graph, "U1001", "exit-draft", "不预约了")
        assert exited["result"]["status"] == "cancelled"
        assert exited["short_term_memory"]["current_task"] is None
        assert exited["short_term_memory"]["suspended_tasks"] == []
    finally:
        graph.close()
