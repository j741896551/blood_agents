"""作品集预约生命周期：只验证 Mock 回执和编排，不把结果当公司旧接口验证。"""

import sqlite3

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from preview_reply import PreviewReplyModel

from blood_agents.llm.reply_composer import _validate_answer
from blood_agents.log.trace_logging import mask_sensitive
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


def _graph(repo, api):
    return ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver(),
    ))


def _ask(graph, text, *, user="U1001", session="lifecycle"):
    return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": user},
                        config={"configurable": {"thread_id": session}})


def test_unavailable_booking_preview_cannot_be_confirmed():
    """模型没把预览送到用户眼前时，确认短句绝不能触发提交。"""
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(), legacy_api=api, checkpointer=MemorySaver(),
    ))
    day = api.site_slots[0]["slots"][0]["date"]
    preview = _ask(graph, f"预约 {day} 上午到市中心献血屋", session="no-booking-preview")
    assert preview["result"]["status"] == "preview_unavailable"
    assert preview["slots"]["status"] == "collecting"
    assert not preview["slots"].get("preview_delivered")
    confirmed = _ask(graph, "确认预约", session="no-booking-preview")
    assert confirmed["result"]["status"] == "preview_unavailable"
    assert all(item["tool"] != "submit_appointment" for item in confirmed["tool_trace"])
    assert repo.list_appointments("U1001") == []


@pytest.mark.parametrize("operation,confirm_word", [
    ("cancel_appointment", "确认取消"),
    ("checkin_appointment", "确认签到"),
])
def test_unavailable_action_preview_cannot_change_submitted_record(operation, confirm_word):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(), legacy_api=api, checkpointer=MemorySaver(),
    ))
    slot = api.site_slots[0]["slots"][0]
    receipt = api.submit_appointment({
        "user_id": "U1001", "site": "市中心献血屋", "date": slot["date"],
        "time_slot": slot["time_slot"], "slot_id": slot["slot_id"],
        "donor": {"kind": "self", "name": "小明", "mobile_masked": "138****5678"},
    }, f"no-{operation}-preview")
    assert receipt["success"]
    request = (f"帮我取消 {slot['date']} 上午市中心献血屋的预约"
               if operation == "cancel_appointment"
               else f"帮我签到 {slot['date']} 上午市中心献血屋的预约")
    preview = _ask(graph, request, session=f"no-{operation}-preview")
    assert preview["result"]["status"] == "preview_unavailable"
    assert "pending_operation" not in preview["slots"]
    confirmed = _ask(graph, confirm_word, session=f"no-{operation}-preview")
    assert all(item["tool"] != operation for item in confirmed["tool_trace"])
    assert repo.list_appointments("U1001")[0].status == "booked"


def test_four_seeded_states_and_cross_user_transition_rejections(tmp_path):
    repo = BloodRepository(str(tmp_path / "states.db"))
    repo.seed_demo_data(include_appointment_samples=True)
    api = MockLegacyBloodAPI(repo)
    records = api.get_user_appointments("U1002")["data"]["appointments"]
    assert {item["status"] for item in records} == {"booked", "cancelled", "checked_in", "donated"}
    assert api.get_user_appointments("U1001")["data"]["appointments"] == []
    for item in records:
        assert item["donor_gender"] == "女"
        assert item["donor_blood_type"] == "O"
        assert item["donation_type"] == "全血"
        assert item["donation_amount"] == 300
        assert "13900001234" not in str(item)
        assert "310101199202021234" not in str(item)
    assert api.cancel_appointment("U1001", "demo-booked")["code"] == "NOT_FOUND"
    assert api.cancel_appointment("U1002", "demo-checked-in")["code"] == "INVALID_STATUS"
    assert api.cancel_appointment("U1002", "demo-donated")["code"] == "INVALID_STATUS"
    first = api.checkin_appointment("U1002", "demo-booked")
    again = api.checkin_appointment("U1002", "demo-booked")
    assert first["success"] and first["data"]["status"] == "checked_in"
    assert again["success"] and again["duplicate"] is True
    assert api.cancel_appointment("U1002", "demo-booked")["code"] == "INVALID_STATUS"
    reopened = MockLegacyBloodAPI(BloodRepository(str(tmp_path / "states.db")))
    assert any(item["appointment_id"] == "demo-booked" and item["status"] == "checked_in"
               for item in reopened.get_user_appointments("U1002")["data"]["appointments"])


def test_graph_cancel_and_checkin_require_explicit_confirmation():
    repo = BloodRepository()
    repo.seed_demo_data(include_appointment_samples=True)
    api = MockLegacyBloodAPI(repo)
    graph = _graph(repo, api)
    day = api.site_slots[0]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到市中心献血屋")
    booked = _ask(graph, "确认预约")
    appointment_id = booked["result"]["data"]["appointment_id"]
    preview = _ask(graph, f"帮我取消 {day} 上午市中心献血屋的预约")
    assert preview["result"]["status"] == "awaiting_action_confirmation"
    assert preview["active_agent"] == "appointment"
    assert all(not item["mutating"] for item in preview["tool_trace"])
    assert repo.list_appointments("U1001")[0].status == "booked"
    wrong_confirmation = _ask(graph, "确认预约")
    assert wrong_confirmation["result"]["status"] == "awaiting_action_confirmation"
    assert repo.list_appointments("U1001")[0].status == "booked"
    confirmed = _ask(graph, "确认取消")
    assert confirmed["result"]["operation"] == "cancel_appointment"
    assert confirmed["result"]["data"]["appointment_id"] == appointment_id
    assert repo.list_appointments("U1001")[0].status == "cancelled"
    assert _ask(graph, "查询我的预约记录")["result"]["appointments"][0]["status"] == "cancelled"

    checkin = _ask(graph, "帮我签到后天上午大学城献血车的预约", user="U1002", session="checkin")
    assert checkin["result"]["status"] == "awaiting_action_confirmation"
    assert _ask(graph, "确认签到", user="U1002", session="checkin")["result"]["data"]["status"] == "checked_in"
    rejected = _ask(graph, "取消后天上午大学城献血车的预约", user="U1002", session="again")
    assert rejected["result"]["code"] == "INVALID_STATUS"
    assert all(not item["mutating"] for item in rejected["tool_trace"])



def test_other_donor_details_required_and_public_data_masked():
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = _graph(repo, api)
    day = api.site_slots[1]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到大学城献血车")
    missing = _ask(graph, "不是我献血，献血人姓名：张三，手机号：13912345678")
    assert "本次献血人身份证号" in missing["result"]["missing"]
    assert "献血类型（全血/机采）" in missing["result"]["missing"]
    preview = _ask(graph, "身份证号：310101199303031234，性别：男，A型血，全血300cc")
    assert preview["result"]["status"] == "awaiting_confirmation"
    booked = _ask(graph, "确认预约")
    assert booked["result"]["success"] is True
    item = booked["result"]["data"]
    assert item["donor_name"] == "张三"
    assert item["donation_amount"] == 300
    assert item["donor_id_card_masked"] == "310***********1234"
    assert "310101199303031234" not in str(booked["result"])
    assert "13912345678" not in str(booked["result"])
    assert mask_sensitive("身份证号310101199303031234，手机13912345678") == (
        "身份证号310***********1234，手机139****5678"
    )


def test_api_and_gradio_callbacks_expose_lifecycle_without_raw_identity(tmp_path, monkeypatch):
    from blood_agents import api as api_module
    from blood_agents.web import ChatApplication

    repo = BloodRepository(str(tmp_path / "api.db"))
    repo.seed_demo_data(include_appointment_samples=True)
    mock = MockLegacyBloodAPI(repo)
    monkeypatch.setattr(api_module, "graph", _graph(repo, mock))
    client = TestClient(api_module.app)
    response = client.post("/chat", json={"user_id": "U1002", "session_id": "api-checkin",
                                          "message": "查询我的预约记录"})
    assert response.status_code == 200
    data = response.json()
    assert {item["status"] for item in data["result"]["appointments"]} == {
        "booked", "cancelled", "checked_in", "donated",
    }
    assert "13900001234" not in response.text
    assert "310101199202021234" not in response.text

    chat = ChatApplication(str(tmp_path / "ui.db"))
    _, history, _, details = chat.respond("查询我的预约记录", [], "U1002", "ui-query")
    assert details["result"]["status"] == "appointments_found"
    assert len(details["result"]["appointments"]) == 4
    assert "13900001234" not in str(details)
    assert "310101199202021234" not in str(details)
    assert history[-1][0] == "查询我的预约记录"


def test_reply_fact_gate_distinguishes_queried_states_from_new_writes():
    result = {"status": "appointments_found", "appointments": [{
        "appointment_id": "abcd1234abcd", "status": "cancelled",
    }]}
    assert _validate_answer("预约号 abcd1234abcd 的状态已取消。", (), "appointment", result)
    with pytest.raises(TypeError):
        _validate_answer("这笔预约已取消。", (), "appointment",
                         {"status": "awaiting_action_confirmation", "operation": "cancel_appointment"})
    with pytest.raises(TypeError):
        _validate_answer("签到成功。", (), "appointment",
                         {"status": "awaiting_action_confirmation", "operation": "checkin_appointment"})


def test_existing_sqlite_appointment_survives_additive_migration(tmp_path):
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE users(user_id TEXT PRIMARY KEY, name TEXT NOT NULL, points INTEGER NOT NULL)")
        conn.execute("INSERT INTO users VALUES('U1001','小明',0)")
        conn.execute("CREATE TABLE appointments(appointment_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                     "site TEXT NOT NULL, appointment_date TEXT NOT NULL, time_slot TEXT NOT NULL, "
                     "status TEXT NOT NULL, idempotency_key TEXT UNIQUE NOT NULL)")
        conn.execute("INSERT INTO appointments VALUES('old-001','U1001','市中心献血屋',"
                     "'2026-10-12','上午','booked','old-key')")
    repo = BloodRepository(path)
    old = repo.list_appointments("U1001")
    assert len(old) == 1 and old[0].appointment_id == "old-001"
    assert old[0].donor_name == "" and old[0].donation_amount == 0
    assert repo.transition_appointment("U1001", "old-001", "cancelled")[0] == "OK"
    assert BloodRepository(path).list_appointments("U1001")[0].status == "cancelled"


def test_model_reply_uses_mock_receipts_for_cancel_checkin_and_four_state_query(monkeypatch):
    import json

    class ReceiptReplyModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            text = "、".join(facts["required_facts"])
            result = facts["result"]
            if result.get("operation") == "cancel_appointment" and result.get("success"):
                text += "，预约已取消。"
            elif result.get("operation") == "checkin_appointment" and result.get("success"):
                text += "，签到成功。"
            elif facts.get("result_success") and facts["result_status"] is None:
                text += "，预约成功。"
            return AIMessage(content=text or "暂无结果")

    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "passthrough")
    repo = BloodRepository()
    repo.seed_demo_data(include_appointment_samples=True)
    api = MockLegacyBloodAPI(repo)
    model = ReceiptReplyModel()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=model, risk_llm=False), legacy_api=api,
        checkpointer=MemorySaver(),
    ))
    queried = _ask(graph, "查询我的预约记录", user="U1002", session="model-query")
    assert queried["response_info"]["mode"] == "llm"
    assert all(label in queried["messages"][-1].content for label in
               ("已预约", "已取消", "已签到", "已献血"))
    _ask(graph, "帮我签到后天上午大学城献血车的预约", user="U1002", session="model-checkin")
    checkin = _ask(graph, "确认签到", user="U1002", session="model-checkin")
    assert checkin["response_info"]["mode"] == "llm"
    assert "签到成功" in checkin["messages"][-1].content
    assert checkin["result"]["data"]["status"] == "checked_in"
    day = api.site_slots[0]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到市中心献血屋", session="model-cancel")
    _ask(graph, "确认预约", session="model-cancel")
    _ask(graph, f"取消 {day} 上午市中心献血屋的预约", session="model-cancel")
    cancelled = _ask(graph, "确认取消", session="model-cancel")
    assert cancelled["response_info"]["mode"] == "llm"
    assert "预约已取消" in cancelled["messages"][-1].content
    assert cancelled["result"]["data"]["status"] == "cancelled"
