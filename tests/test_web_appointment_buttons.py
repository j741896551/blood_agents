"""预约按钮复用原 LangGraph 流程；旧按钮不能越过服务端确认门禁。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from langgraph.checkpoint.memory import MemorySaver
from preview_reply import PreviewReplyModel

from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_conversational_graph, build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.web import ChatApplication, build_ui
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


@pytest.fixture()
def button_app(tmp_path):
    app = ChatApplication(str(tmp_path / "button-demo.db"))
    app.graph.close()
    app.business_repo = BloodRepository(app.repo.db_path)
    app.business_repo.seed_demo_data(include_appointment_samples=True)
    router = IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False)
    app.graph = ConversationGraph(
        build_graph(app.repo, router, legacy_api=MockLegacyBloodAPI(app.business_repo),
                    checkpointer=MemorySaver()),
        conversation_store=app.repo,
    )
    yield app
    app.graph.close()


def _preview(app, session="button-session"):
    day = MockLegacyBloodAPI(app.business_repo).site_slots[0]["slots"][0]["date"]
    return app.respond(f"预约 {day} 上午到市中心献血屋", [], "U1001", session)


def test_booking_button_confirms_once_and_duplicate_is_stale(button_app):
    app = button_app
    preview = _preview(app)
    pending = preview[3]["pending_action"]
    assert pending["operation"] == "book"
    assert pending["confirm_label"] == "确认预约"
    assert preview[3]["slots"]["preview_delivered"] is True
    assert app.graph.get_pending_appointment_button("U1002", "button-session") is None
    assert app.graph.get_pending_appointment_button("U1001", "other-session") is None
    assert app.restore_history("U1001", "button-session")[1]["pending_action"] == pending

    confirmed = app.confirm_pending_button(preview[1], "U1001", "button-session", preview[3])
    assert confirmed[3]["result"]["success"] is True
    assert [item["tool"] for item in confirmed[3]["tool_trace"]].count("submit_appointment") == 1
    assert confirmed[3]["pending_action"] is None
    assert len(app.business_repo.list_appointments("U1001")) == 1

    repeated = app.confirm_pending_button(confirmed[1], "U1001", "button-session", preview[3])
    assert repeated[3]["ui_action"]["status"] == "stale"
    assert len(app.business_repo.list_appointments("U1001")) == 1


def test_old_button_after_change_or_user_switch_never_submits(button_app):
    app = button_app
    preview = _preview(app, "change-session")
    old = preview[3]
    changed = app.respond("预约改成下午", preview[1], "U1001", "change-session")
    assert changed[3]["pending_action"]["token"] != old["pending_action"]["token"]
    stale = app.confirm_pending_button(changed[1], "U1001", "change-session", old)
    assert stale[3]["ui_action"]["status"] == "stale"
    cross_user = app.confirm_pending_button(preview[1], "U1002", "change-session", old)
    assert cross_user[3]["ui_action"]["status"] == "stale"
    cross_session = app.confirm_pending_button(preview[1], "U1001", "other-session", old)
    assert cross_session[3]["ui_action"]["status"] == "stale"
    assert app.business_repo.list_appointments("U1001") == []
    assert app.business_repo.list_appointments("U1002")  # 仅有原有演示记录


def test_tampered_button_operation_or_message_is_rejected(button_app):
    app = button_app
    preview = _preview(app, "tamper-session")
    wrong_operation = deepcopy(preview[3])
    wrong_operation["pending_action"]["operation"] = "cancel_appointment"
    wrong_message = deepcopy(preview[3])
    wrong_message["pending_action"]["confirm_text"] = "确认取消"
    for details in (wrong_operation, wrong_message):
        rejected = app.confirm_pending_button(preview[1], "U1001", "tamper-session", details)
        assert rejected[3]["ui_action"]["status"] == "stale"
    assert app.business_repo.list_appointments("U1001") == []


def test_abandon_button_exits_draft_without_cancelling_submitted_record(button_app):
    app = button_app
    preview = _preview(app, "abandon-session")
    abandoned = app.abandon_pending_button(preview[1], "U1001", "abandon-session", preview[3])
    assert abandoned[3]["result"]["scope"] == "draft"
    assert abandoned[3]["pending_action"] is None
    assert all(item["tool"] != "submit_appointment" for item in abandoned[3]["tool_trace"])
    assert app.business_repo.list_appointments("U1001") == []
    stale = app.confirm_pending_button(abandoned[1], "U1001", "abandon-session", preview[3])
    assert stale[3]["ui_action"]["status"] == "stale"


def test_existing_appointment_cancel_button_preserves_original_record_until_confirm(button_app):
    app = button_app
    preview = _preview(app, "cancel-session")
    booked = app.confirm_pending_button(preview[1], "U1001", "cancel-session", preview[3])
    appointment_id = booked[3]["result"]["data"]["appointment_id"]
    requested = app.respond(
        f"取消预约号 {appointment_id} 的预约", booked[1], "U1001", "cancel-session",
    )
    pending = requested[3]["pending_action"]
    assert pending["operation"] == "cancel_appointment"
    assert pending["confirm_text"] == "确认取消"
    assert app.business_repo.list_appointments("U1001")[0].status == "booked"

    abandoned = app.abandon_pending_button(requested[1], "U1001", "cancel-session", requested[3])
    assert abandoned[3]["result"]["status"] == "action_aborted"
    assert app.business_repo.list_appointments("U1001")[0].status == "booked"
    requested_again = app.respond(
        f"取消预约号 {appointment_id} 的预约", abandoned[1], "U1001", "cancel-session",
    )
    cancelled = app.confirm_pending_button(
        requested_again[1], "U1001", "cancel-session", requested_again[3],
    )
    assert cancelled[3]["result"]["success"] is True
    assert [item["tool"] for item in cancelled[3]["tool_trace"]].count("cancel_appointment") == 1
    assert app.business_repo.list_appointments("U1001")[0].status == "cancelled"


def test_checkin_button_uses_existing_booked_record(button_app):
    app = button_app
    booked = next(item for item in app.business_repo.list_appointments("U1002") if item.status == "booked")
    preview = app.respond(
        f"签到预约号 {booked.appointment_id}", [], "U1002", "checkin-session",
    )
    assert preview[3]["pending_action"]["operation"] == "checkin_appointment"
    assert any(item.appointment_id == booked.appointment_id and item.status == "booked"
               for item in app.business_repo.list_appointments("U1002"))
    checked = app.confirm_pending_button(preview[1], "U1002", "checkin-session", preview[3])
    assert checked[3]["result"]["success"] is True
    assert [item["tool"] for item in checked[3]["tool_trace"]].count("checkin_appointment") == 1
    assert any(item.appointment_id == booked.appointment_id and item.status == "checked_in"
               for item in app.business_repo.list_appointments("U1002"))


def test_concurrent_double_click_accepts_only_one_button_version(button_app):
    app = button_app
    preview = _preview(app, "double-click")

    def click():
        return app.confirm_pending_button(preview[1], "U1001", "double-click", preview[3])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: click(), range(2)))
    assert sum(bool(result[3].get("result", {}).get("success")) for result in results) == 1
    assert sum(result[3].get("ui_action", {}).get("status") == "stale" for result in results) == 1
    assert len(app.business_repo.list_appointments("U1001")) == 1


def test_file_checkpoint_restores_button_version_without_replaying_history(button_app):
    app = button_app
    router = IntentRouter(response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False)
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    preview = _preview(app, "persistent-button")
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    history, details, _ = app.restore_history("U1001", "persistent-button")
    assert history == preview[1]
    assert details["pending_action"]["token"] == preview[3]["pending_action"]["token"]
    confirmed = app.confirm_pending_button(history, "U1001", "persistent-button", details)
    assert confirmed[3]["result"]["success"] is True
    assert len(app.business_repo.list_appointments("U1001")) == 1


def test_no_reply_model_never_exposes_confirm_button(button_app):
    app = button_app
    app.graph = ConversationGraph(
        build_graph(app.repo, IntentRouter(), checkpointer=MemorySaver()),
        conversation_store=app.repo,
    )
    preview = _preview(app, "no-model-button")
    assert preview[3]["result"]["status"] == "preview_unavailable"
    assert preview[3]["pending_action"] is None
    assert app.pending_button_updates(preview[3])[0]["visible"] is False
    assert app.restore_history("U1001", "no-model-button")[1].get("pending_action") is None


def test_button_visibility_comes_from_structured_state_and_page_callbacks(button_app):
    app = button_app
    hidden = app.pending_button_updates({})
    assert all(item["visible"] is False for item in hidden)
    preview = _preview(app, "visible-session")
    shown = app.pending_button_updates(preview[3])
    assert all(item["visible"] is True for item in shown)
    assert "点击「确认预约」" in shown[0]["value"]
    assert shown[1]["value"] == "确认预约"
    demo = build_ui(app)
    assert demo.get_api_info()
    assert any(getattr(block.fn, "__func__", None) is ChatApplication.confirm_pending_button
               for block in demo.fns.values())
    assert any(getattr(block.fn, "__func__", None) is ChatApplication.abandon_pending_button
               for block in demo.fns.values())
