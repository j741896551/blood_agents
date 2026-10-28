"""积分待确认按钮只改变交互入口，不绕过原图的写操作与回执。"""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from test_points_extended import FactsReplyModel, UnderstandingModel

from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_conversational_graph, build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.web import ChatApplication, build_ui
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


@pytest.fixture()
def points_app(tmp_path):
    app = ChatApplication(str(tmp_path / "points-button.db"))
    app.graph.close()
    app.business_repo = BloodRepository(app.repo.db_path)
    app.business_repo.seed_demo_data(include_appointment_samples=True)
    router = IntentRouter(llm=UnderstandingModel(), response_llm=FactsReplyModel(),
                          risk_llm=False, evidence_llm=False)
    app.graph = ConversationGraph(
        build_graph(app.repo, router, legacy_api=MockLegacyBloodAPI(app.business_repo),
                    checkpointer=MemorySaver()),
        conversation_store=app.repo,
    )
    yield app, router
    app.graph.close()


def _address(app, user="U1001", recipient="张三", mobile="13800001234"):
    return app.business_repo.save_address(user, recipient, mobile, "上海市人民路100号")


def test_redeem_button_confirms_once_and_old_click_is_stale(points_app):
    app, _ = points_app
    _address(app)
    preview = app.respond("兑换 R001", [], "U1001", "redeem-button")
    pending = preview[3]["pending_action"]
    assert pending["operation"] == "points:redeem"
    assert pending["confirm_label"] == "确认兑换"
    assert app.graph.get_pending_appointment_button("U1001", "redeem-button") is None
    assert app.business_repo.get_points("U1001") == 1200

    confirmed = app.confirm_pending_button(preview[1], "U1001", "redeem-button", preview[3])
    assert confirmed[3]["result"]["status"] == "redeemed"
    assert [item["tool"] for item in confirmed[3]["tool_trace"]].count("redeem_reward") == 1
    assert app.business_repo.get_points("U1001") == 700
    assert confirmed[3]["pending_action"] is None

    repeated = app.confirm_pending_button(confirmed[1], "U1001", "redeem-button", preview[3])
    assert repeated[3]["ui_action"]["status"] == "stale"
    assert app.business_repo.get_points("U1001") == 700


def test_address_save_button_then_separate_redemption_button(points_app):
    app, _ = points_app
    need_address = app.respond("兑换 R001", [], "U1001", "address-then-redeem")
    assert need_address[3]["pending_action"] is None
    preview = app.respond(
        "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号",
        need_address[1], "U1001", "address-then-redeem",
    )
    assert preview[3]["pending_action"]["operation"] == "points:address_add"
    assert app.business_repo.list_addresses("U1001") == []
    assert app.business_repo.get_points("U1001") == 1200

    saved = app.confirm_pending_button(preview[1], "U1001", "address-then-redeem", preview[3])
    assert [item["tool"] for item in saved[3]["tool_trace"]].count("save_shipping_address") == 1
    assert saved[3]["pending_action"]["operation"] == "points:redeem"
    assert saved[3]["pending_action"]["token"] != preview[3]["pending_action"]["token"]
    assert app.business_repo.get_points("U1001") == 1200

    stale = app.confirm_pending_button(saved[1], "U1001", "address-then-redeem", preview[3])
    assert stale[3]["ui_action"]["status"] == "stale"
    redeemed = app.confirm_pending_button(saved[1], "U1001", "address-then-redeem", saved[3])
    assert redeemed[3]["result"]["status"] == "redeemed"
    assert app.business_repo.get_points("U1001") == 700


def test_cancel_order_abandon_then_confirm_uses_original_order(points_app):
    app, _ = points_app
    preview = app.respond("取消订单 demo-order-3", [], "U1002", "cancel-order-button")
    assert preview[3]["pending_action"]["operation"] == "points:cancel_order"
    order = lambda: next(item for item in app.business_repo.list_redemptions("U1002")
                         if item["redemption_id"] == "demo-order-3")
    assert order()["status"] == "待发货"

    abandoned = app.abandon_pending_button(preview[1], "U1002", "cancel-order-button", preview[3])
    assert abandoned[3]["result"]["status"] == "redemption_aborted"
    assert order()["status"] == "待发货"
    again = app.respond("取消订单 demo-order-3", abandoned[1], "U1002", "cancel-order-button")
    cancelled = app.confirm_pending_button(again[1], "U1002", "cancel-order-button", again[3])
    assert cancelled[3]["result"]["status"] == "cancel_order_completed"
    assert [item["tool"] for item in cancelled[3]["tool_trace"]].count("cancel_redemption_order") == 1
    assert order()["status"] == "已取消"


@pytest.mark.parametrize("operation", ["address_update", "address_default", "order_address"])
def test_address_and_order_buttons_follow_existing_write_tools(points_app, operation):
    app, _ = points_app
    first = _address(app)
    second = app.business_repo.save_address("U1001", "李四", "13900004321", "上海市大学路8号")
    if operation == "address_update":
        message = f"修改收货地址 {first['address_id']}，收货人张三，手机号13800001234，详细地址上海市大学路8号"
        expected_status, tool = "address_update_completed", "save_shipping_address"
    elif operation == "address_default":
        message = f"设置默认收货地址 {second['address_id']}"
        expected_status, tool = "address_default_completed", "set_default_shipping_address"
    else:
        receipt = app.business_repo.redeem("U1001", "R001", "order-button-setup", first["address_id"])
        message = f"填写订单收货信息 {receipt['redemption_id']} 地址 {second['address_id']}"
        expected_status, tool = "order_address_completed", "set_order_shipping_address"
    preview = app.respond(message, [], "U1001", f"{operation}-button")
    assert preview[3]["pending_action"]["operation"] == f"points:{operation}"
    confirmed = app.confirm_pending_button(preview[1], "U1001", f"{operation}-button", preview[3])
    assert confirmed[3]["result"]["status"] == expected_status
    assert [item["tool"] for item in confirmed[3]["tool_trace"]].count(tool) == 1


def test_old_points_button_isolated_by_user_session_and_operation(points_app):
    app, _ = points_app
    _address(app)
    preview = app.respond("兑换 R001", [], "U1001", "isolation-points")
    tampered = deepcopy(preview[3])
    tampered["pending_action"]["operation"] = "points:cancel_order"
    for user, session, details in (
        ("U1002", "isolation-points", preview[3]),
        ("U1001", "other-points-session", preview[3]),
        ("U1001", "isolation-points", tampered),
    ):
        rejected = app.confirm_pending_button(preview[1], user, session, details)
        assert rejected[3]["ui_action"]["status"] == "stale"
    assert app.business_repo.get_points("U1001") == 1200


def test_points_button_double_click_only_writes_once(points_app):
    app, _ = points_app
    _address(app)
    preview = app.respond("兑换 R001", [], "U1001", "points-double-click")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: app.confirm_pending_button(
            preview[1], "U1001", "points-double-click", preview[3],
        ), range(2)))
    assert sum(item[3].get("result", {}).get("status") == "redeemed" for item in results) == 1
    assert sum(item[3].get("ui_action", {}).get("status") == "stale" for item in results) == 1
    assert app.business_repo.get_points("U1001") == 700


def test_redacted_preview_does_not_restore_button_after_sqlite_graph_rebuild(points_app):
    app, router = points_app
    _address(app)
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    preview = app.respond("兑换 R001", [], "U1001", "persistent-points-button")
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    history, details, _ = app.restore_history("U1001", "persistent-points-button")
    assert history[-1][1] == "[涉及个人资料，消息内容未保存]"
    assert details.get("pending_action") is None
    assert app.graph.get_pending_button("U1001", "persistent-points-button")["token"] == (
        preview[3]["pending_action"]["token"]
    )
    assert app.business_repo.get_points("U1001") == 1200


def test_visible_cancel_order_preview_restores_button_after_graph_rebuild(points_app):
    app, router = points_app
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    preview = app.respond("取消订单 demo-order-3", [], "U1002", "visible-points-restore")
    app.graph.close()
    app.graph = build_conversational_graph(app.repo, router=router)
    history, details, _ = app.restore_history("U1002", "visible-points-restore")
    assert history == preview[1]
    assert details["pending_action"]["token"] == preview[3]["pending_action"]["token"]
    cancelled = app.confirm_pending_button(history, "U1002", "visible-points-restore", details)
    assert cancelled[3]["result"]["status"] == "cancel_order_completed"


def test_points_button_invalid_after_another_preview_replaces_it(points_app):
    app, _ = points_app
    _address(app)
    old = app.respond("兑换 R001", [], "U1001", "points-replaced")
    changed = app.respond("设置默认收货地址 " + app.business_repo.list_addresses("U1001")[0]["address_id"],
                          old[1], "U1001", "points-replaced")
    assert changed[3]["pending_action"]["operation"] == "points:address_default"
    assert changed[3]["pending_action"]["token"] != old[3]["pending_action"]["token"]
    stale = app.confirm_pending_button(changed[1], "U1001", "points-replaced", old[3])
    assert stale[3]["ui_action"]["status"] == "stale"
    assert app.business_repo.get_points("U1001") == 1200


def test_points_without_valid_model_preview_has_no_button(points_app):
    app, _ = points_app
    _address(app)
    app.graph = ConversationGraph(
        build_graph(app.repo, IntentRouter(), checkpointer=MemorySaver()),
        conversation_store=app.repo,
    )
    preview = app.respond("兑换 R001", [], "U1001", "points-no-model")
    assert preview[3]["pending_action"] is None
    assert app.graph.get_pending_button("U1001", "points-no-model") is None
    assert app.business_repo.get_points("U1001") == 1200


def test_buttons_live_in_chat_card_and_update_before_title_model(points_app):
    app, _ = points_app
    demo = build_ui(app)
    components = {item["id"]: item for item in demo.config["components"]}
    chat_card = next(item for item in components.values()
                     if item.get("props", {}).get("elem_id") == "conversation-card")

    def find_layout(node, target):
        if node["id"] == target:
            return node
        return next((found for child in node.get("children", [])
                     if (found := find_layout(child, target))), None)

    card_layout = find_layout(demo.config["layout"], chat_card["id"])
    assert components[card_layout["children"][0]["id"]]["type"] == "chatbot"
    assert components[card_layout["children"][1]["id"]]["props"]["elem_id"] == "conversation-actions"
    dependencies = {item["id"]: item for item in demo.config["dependencies"]}
    for task_id, block in demo.fns.items():
        if getattr(block.fn, "__name__", "") not in {
            "respond_with_stream", "confirm_pending_button", "abandon_pending_button",
        }:
            continue
        update = next(item for item in dependencies.values() if item.get("trigger_after") == task_id)
        title = next(item for item in dependencies.values() if item.get("trigger_after") == update["id"])
        assert demo.fns[update["id"]].fn.__name__ == "pending_button_updates"
        assert demo.fns[title["id"]].fn.__name__ == "refresh_session_title"


def test_gradio_reply_mentions_button_without_promising_it_to_api(points_app):
    app, _ = points_app

    class ButtonAwareReply(FactsReplyModel):
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            facts = json.loads(messages[-1].content)
            if facts.get("ui_confirmation") and facts.get("revision_feedback"):
                return AIMessage(content=("订单 demo-order-3 当前待发货，尚未取消。可点击“确认取消订单”按钮，"
                                          "也可以回复“确认取消订单”后再提交。"))
            return super().invoke(messages)

    reply = ButtonAwareReply()
    router = IntentRouter(llm=UnderstandingModel(), response_llm=reply,
                          risk_llm=False, evidence_llm=False)
    app.graph = ConversationGraph(build_graph(app.repo, router, checkpointer=MemorySaver()),
                                  conversation_store=app.repo)
    preview = app.respond("取消订单 demo-order-3", [], "U1002", "button-wording-ui")
    assert preview[3]["pending_action"]["confirm_label"] == "确认取消订单"
    assert "点击" in preview[1][-1][1] and "按钮" in preview[1][-1][1]
    assert len(reply.calls) == 2
    assert json.loads(reply.calls[0][-1].content)["ui_confirmation"]["confirm_label"] == "确认取消订单"
    assert "Gradio" in reply.calls[0][0].content

    reply.calls.clear()
    app.graph.invoke(
        {"user_id": "U1002", "messages": [HumanMessage(content="取消订单 demo-order-1")]},
        {"configurable": {"thread_id": "button-wording-api"}, "metadata": {"entrypoint": "api"}},
    )
    assert len(reply.calls) == 1
    assert "ui_confirmation" not in json.loads(reply.calls[0][-1].content)
