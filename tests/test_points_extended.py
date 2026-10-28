"""积分订单/地址的多轮图级回归：可控语义模型，真实 Mock SQLite 写回执。"""

import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents import api as api_module
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.web import ChatApplication
from local_business.repository import BloodRepository


class UnderstandingModel:
    def invoke(self, messages):
        text = json.loads(messages[-1].content)["message"]
        if "主导航" in messages[0].content:
            return AIMessage(content='{"intent":"points"}')
        action = "query"
        fields = {}
        if text.startswith("确认"):
            action = "confirm"
        elif "取消订单" in text:
            action = "cancel_order"
        elif "设置默认" in text:
            action = "address_default"
        elif "填写订单收货信息" in text:
            action = "order_address"
        elif "修改收货地址" in text:
            action = "address_update"
        elif "新增收货地址" in text:
            action = "address_add"
        elif "收货地址" in text:
            action = "addresses"
        elif "订单" in text or "兑换记录" in text:
            action = "orders"
        elif "推荐" in text:
            action = "recommend"
        elif "兑换" in text:
            action = "redeem"
        for pattern, key in ((r"R\d{3}", "reward_id"), (r"A[0-9A-F]{8}", "address_id"),
                             (r"(?:demo-order-\d+|[0-9a-f]{12})", "order_id")):
            match = re.search(pattern, text)
            if match:
                fields[key] = match.group(0)
        if "张三" in text:
            fields["recipient"] = "张三"
        if "李四" in text:
            fields["recipient"] = "李四"
        for detail in ("上海市人民路100号", "上海市大学路8号"):
            if detail in text:
                fields["address_detail"] = detail
        return AIMessage(content=json.dumps({"action": action, **fields}, ensure_ascii=False))


class MisroutedShippingModel(UnderstandingModel):
    """复现真实模型把待确认兑换的新地址错判为旧订单地址修改。"""

    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        text = payload["message"]
        if "积分子智能体" in messages[0].content and "收货人 王五" in text:
            return AIMessage(content=json.dumps({
                "action": "order_address", "recipient": "王五",
                "address_detail": "深圳市演示路7号楼5楼",
            }, ensure_ascii=False))
        return super().invoke(messages)


class FactsReplyModel:
    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        status = facts["result_status"]
        if status == "redeemed":
            return AIMessage(content="兑换成功：" + "、".join(facts["required_facts"]))
        if status == "orders_found":
            columns = facts["orders_table_columns"]
            rows = ["| " + " | ".join(columns) + " |",
                    "| " + " | ".join("---" for _ in columns) + " |"]
            keys = ("redemption_id", "reward", "points_cost", "status", "recipient",
                    "mobile_masked", "address_detail", "created_at", "paid_at", "shipped_at",
                    "delivered_at", "cancelled_at")
            for order in facts["result"]["orders"]:
                rows.append("| " + " | ".join(
                    str(order.get(key) or facts["orders_missing_value"]) for key in keys
                ) + " |")
            return AIMessage(content="\n".join(rows))
        prefix = "尚未执行，等待确认：" if status in {
            "awaiting_redemption_confirmation", "awaiting_points_confirmation",
        } else "尚未下单，请填写收货地址：" if status == "redemption_address_required" else "请选收货地址：" if status == "address_selection_required" else "暂无相关数据：" if status in {
            "no_orders", "no_addresses", "no_recommendations", "no_pending_redemption",
        } else "查询与操作预览："
        return AIMessage(content=prefix + "、".join(facts["required_facts"]))


def setup(understanding=None):
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=understanding or UnderstandingModel(),
                           response_llm=FactsReplyModel(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    return repo, graph


def ask(graph, text, user="U1001", session="points-extended"):
    return graph.invoke({"user_id": user, "messages": [HumanMessage(content=text)]},
                        {"configurable": {"thread_id": session}})


def test_address_order_lifecycle_and_cancellation_receipt():
    repo, graph = setup()
    address_preview = ask(graph, "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号")
    assert address_preview["result"]["status"] == "awaiting_points_confirmation"
    assert repo.list_addresses("U1001") == []
    assert "138****1234" in address_preview["messages"][-1].content
    saved = ask(graph, "确认保存地址")
    assert saved["result"]["status"] == "address_add_completed"
    address_id = saved["result"]["data"]["address_id"]
    assert saved["result"]["data"]["is_default"] is True
    assert repo.list_addresses("U1001")[0]["mobile"] == "138****1234"

    preview = ask(graph, "兑换 R001")
    assert preview["result"]["status"] == "awaiting_redemption_confirmation"
    assert repo.get_points("U1001") == 1200
    redeemed = ask(graph, "确认兑换")
    order_id = redeemed["result"]["redemption_id"]
    assert redeemed["result"]["order_status"] == "待发货"
    assert repo.get_points("U1001") == 700

    queried = ask(graph, "查询我的兑换订单")
    assert queried["result"]["orders"][0]["redemption_id"] == order_id
    assert queried["result"]["orders"][0]["address_id"] == address_id
    assert queried["result"]["orders"][0]["status"] == "待发货"
    assert all(column in queried["messages"][-1].content for column in (
        "收货人", "收货电话", "收货地址", "下单时间", "付款时间", "发货时间", "签收时间", "取消时间",
    ))
    assert "query_redemption_orders" in [item["tool"] for item in queried["tool_trace"]]

    cancel_preview = ask(graph, f"取消订单 {order_id}")
    assert cancel_preview["result"]["status"] == "awaiting_points_confirmation"
    assert repo.list_redemptions("U1001")[0]["status"] == "待发货"
    cancelled = ask(graph, "确认取消订单")
    assert cancelled["result"]["status"] == "cancel_order_completed"
    assert cancelled["result"]["data"]["status"] == "已取消"
    assert repo.get_points("U1001") == 1200
    assert [item["tool"] for item in cancelled["tool_trace"]] == ["cancel_redemption_order"]


def test_order_list_uses_one_complete_table_before_and_after_cancellation():
    repo, graph = setup()
    session = "stable-order-table"
    first = ask(graph, "查询我的兑换订单", user="U1002", session=session)
    expected_columns = (
        "订单号", "礼品", "积分", "状态", "收货人", "收货电话", "收货地址",
        "下单时间", "付款时间", "发货时间", "签收时间", "取消时间",
    )
    first_answer = first["messages"][-1].content
    assert all(column in first_answer for column in expected_columns)
    assert "演示收货人" in first_answer
    assert "演示省演示市示例区" in first_answer
    assert "demo-order-2" in first_answer
    assert "2026-08-02 10:15:00" in first_answer

    preview = ask(graph, "取消订单 demo-order-1", user="U1002", session=session)
    assert preview["result"]["status"] == "awaiting_points_confirmation"
    cancelled = ask(graph, "确认取消订单", user="U1002", session=session)
    assert cancelled["result"]["status"] == "cancel_order_completed"
    refreshed = ask(graph, "再查我的兑换订单", user="U1002", session=session)
    second_answer = refreshed["messages"][-1].content
    assert all(column in second_answer for column in expected_columns)
    assert "demo-order-1" in second_answer
    assert repo.list_redemptions("U1002")[-1]["cancelled_at"]


def test_short_cancel_confirmation_keeps_pending_order_and_allows_retry():
    class NaturalCancelReply(FactsReplyModel):
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            result = facts["result"]
            if result.get("status") == "awaiting_points_confirmation" and result.get("operation") == "cancel_order":
                order = result.get("order") or {}
                order_id = order.get("redemption_id") or result["order_id"]
                return AIMessage(content=(
                    f"订单 {order_id} 当前待发货，尚未取消。"
                    "请回复‘确认取消订单’后，我才会调用取消接口。"
                ))
            return super().invoke(messages)

    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=UnderstandingModel(), response_llm=NaturalCancelReply(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    session = "short-order-cancel"
    preview = ask(graph, "取消订单 demo-order-3", user="U1002", session=session)
    assert preview["response_info"]["mode"] == "llm"
    assert preview["points_state"]["kind"] == "cancel_order"

    # 未达到预览中的完整确认词时不得写接口，但模型应自然解释并保留已送达的预览。
    incomplete = ask(graph, "确认取消", user="U1002", session=session)
    assert incomplete["result"]["status"] == "awaiting_points_confirmation"
    assert incomplete["response_info"]["mode"] == "llm"
    assert "确认取消订单" in incomplete["messages"][-1].content
    assert incomplete["points_state"]["kind"] == "cancel_order"
    assert incomplete["tool_trace"] == []
    assert next(order for order in repo.list_redemptions("U1002")
                if order["redemption_id"] == "demo-order-3")["status"] == "待发货"

    confirmed = ask(graph, "确认取消订单", user="U1002", session=session)
    assert confirmed["result"]["status"] == "cancel_order_completed"
    assert [item["tool"] for item in confirmed["tool_trace"]] == ["cancel_redemption_order"]
    assert next(order for order in repo.list_redemptions("U1002")
                if order["redemption_id"] == "demo-order-3")["status"] == "已取消"


def test_short_cancel_confirmation_can_be_abandoned_without_write():
    class AbandonUnderstanding(UnderstandingModel):
        def invoke(self, messages):
            if "主导航" not in messages[0].content and json.loads(messages[-1].content)["message"] == "不取消了":
                return AIMessage(content='{"action":"cancel"}')
            return super().invoke(messages)

    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=AbandonUnderstanding(), response_llm=FactsReplyModel(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    session = "abandon-order-cancel"
    ask(graph, "取消订单 demo-order-3", user="U1002", session=session)
    incomplete = ask(graph, "确认取消", user="U1002", session=session)
    assert incomplete["points_state"]["kind"] == "cancel_order"

    abandoned = ask(graph, "不取消了", user="U1002", session=session)
    assert abandoned["points_state"] == {}
    assert abandoned["tool_trace"] == []
    assert next(order for order in repo.list_redemptions("U1002")
                if order["redemption_id"] == "demo-order-3")["status"] == "待发货"


def test_address_update_default_and_order_shipping_info():
    repo, graph = setup()
    first = ask(graph, "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号")
    assert first["result"]["status"] == "awaiting_points_confirmation"
    address_a = ask(graph, "确认保存地址")["result"]["data"]["address_id"]
    ask(graph, "新增收货地址：收货人李四，手机号13900004321，详细地址上海市大学路8号")
    address_b = ask(graph, "确认保存地址")["result"]["data"]["address_id"]
    assert repo.list_addresses("U1001")[0]["address_id"] == address_a
    default_preview = ask(graph, f"设置默认收货地址 {address_b}")
    assert default_preview["result"]["status"] == "awaiting_points_confirmation"
    defaulted = ask(graph, "确认设为默认")
    assert defaulted["result"]["status"] == "address_default_completed"
    assert repo.list_addresses("U1001")[0]["address_id"] == address_b

    edit_preview = ask(graph, f"修改收货地址 {address_a}，收货人张三，手机号13800001234，详细地址上海市大学路8号")
    assert edit_preview["result"]["status"] == "awaiting_points_confirmation"
    edited = ask(graph, "确认保存地址")
    assert edited["result"]["status"] == "address_update_completed"
    assert next(a for a in repo.list_addresses("U1001") if a["address_id"] == address_a)["detail"] == "上海市大学路8号"

    ask(graph, "兑换 R001")
    order_id = ask(graph, "确认兑换")["result"]["redemption_id"]
    change = ask(graph, f"填写订单收货信息 {order_id} 地址 {address_a}")
    assert change["result"]["status"] == "awaiting_points_confirmation"
    updated = ask(graph, "确认填写收货信息")
    assert updated["result"]["status"] == "order_address_completed"
    assert repo.list_redemptions("U1001")[0]["address_id"] == address_a


def test_recommendation_and_mock_statuses_are_data_driven():
    _, graph = setup()
    recommended = ask(graph, "按我的积分推荐礼品")
    assert recommended["result"]["status"] == "recommendations_found"
    assert {r["reward_id"] for r in recommended["result"]["rewards"]} == {"R001", "R002"}
    orders = ask(graph, "查询兑换订单", user="U1002", session="statuses")
    assert {o["status"] for o in orders["result"]["orders"]} == {
        "未支付", "已取消", "待发货", "已发货", "运输中", "已收货",
    }
    blocked = ask(graph, "取消订单 demo-order-5", user="U1002", session="statuses")
    assert blocked["result"]["status"] == "order_cancel_unavailable"
    assert all(item["tool"] != "cancel_redemption_order" for item in blocked["tool_trace"])


def test_address_can_be_collected_across_turns_and_partially_updated():
    repo, graph = setup()
    first = ask(graph, "新增收货地址：收货人张三")
    assert first["result"]["status"] == "address_details_required"
    premature = ask(graph, "确认保存地址")
    assert premature["result"]["status"] == "address_details_required"
    assert repo.list_addresses("U1001") == []
    preview = ask(graph, "手机号13800001234，详细地址上海市人民路100号")
    assert preview["result"]["status"] == "awaiting_points_confirmation"
    address_id = ask(graph, "确认保存地址")["result"]["data"]["address_id"]
    edit = ask(graph, f"修改收货地址 {address_id}，详细地址上海市大学路8号")
    assert edit["result"]["status"] == "awaiting_points_confirmation"
    assert edit["result"]["address"]["mobile"] == "138****1234"
    ask(graph, "确认保存地址")
    assert repo.list_addresses("U1001")[0]["detail"] == "上海市大学路8号"
    assert repo.list_addresses("U1001")[0]["mobile"] == "138****1234"


def test_order_and_address_are_isolated_by_login_user():
    repo, graph = setup()
    ask(graph, "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号")
    own_address = ask(graph, "确认保存地址")["result"]["data"]["address_id"]
    other_addresses = ask(graph, "查看我的收货地址", user="U1002", session="other")
    assert other_addresses["result"]["addresses"] == []
    other_orders = ask(graph, "查询兑换订单", user="U1002", session="other")
    assert len(other_orders["result"]["orders"]) == 6
    own_orders = ask(graph, "查询兑换订单")
    assert own_orders["result"]["orders"] == []
    foreign_order = ask(graph, "查询订单 demo-order-3")
    assert foreign_order["result"]["status"] == "order_not_found"
    assert "demo-order-3" not in foreign_order["messages"][-1].content
    refused = ask(graph, f"设置默认收货地址 {own_address}", user="U1002", session="other")
    assert refused["result"]["status"] == "address_selection_required"
    assert repo.list_addresses("U1001")[0]["is_default"] is True


def test_false_order_cancellation_claim_is_blocked_before_user_sees_it():
    class FalseReply(FactsReplyModel):
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_status"] == "awaiting_points_confirmation":
                return AIMessage(content="订单取消成功；" + "、".join(facts["required_facts"]))
            return super().invoke(messages)

    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=UnderstandingModel(), response_llm=FalseReply(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    state = ask(graph, "取消订单 demo-order-3", user="U1002", session="false-order-cancel")
    assert state["result"]["status"] == "awaiting_points_confirmation"
    assert state["response_info"]["mode"] == "unavailable"
    assert state["points_state"] == {}
    assert next(item for item in repo.list_redemptions("U1002")
                if item["redemption_id"] == "demo-order-3")["status"] == "待发货"


def test_redemption_without_address_needs_shipping_info_and_persists_after_reopen(tmp_path):
    db_path = str(tmp_path / "points.db")
    repo = BloodRepository(db_path)
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=UnderstandingModel(), response_llm=FactsReplyModel(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    first = ask(graph, "兑换 R001", session="persist")
    assert first["result"]["status"] == "redemption_address_required"
    assert first["active_agent"] == "points"
    premature = ask(graph, "确认兑换", session="persist")
    assert premature["result"]["status"] == "address_details_required"
    assert repo.list_redemptions("U1001") == []
    assert repo.get_points("U1001") == 1200
    address_preview = ask(graph, "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号",
                          session="persist")
    assert address_preview["result"]["status"] == "awaiting_points_confirmation"
    saved = ask(graph, "确认保存地址", session="persist")
    assert saved["result"]["status"] == "awaiting_redemption_confirmation"
    assert saved["result"]["address"]["recipient"] == "张三"
    assert saved["result"]["address"]["mobile"] == "138****1234"
    assert "确认兑换" in saved["messages"][-1].content
    assert repo.list_redemptions("U1001") == []
    receipt = ask(graph, "确认兑换", session="persist")
    assert receipt["result"]["order_status"] == "待发货"
    assert receipt["result"]["address"]["mobile"] == "138****1234"
    assert [item["tool"] for item in receipt["tool_trace"]] == ["redeem_reward"]
    order_id = receipt["result"]["redemption_id"]
    reopened = BloodRepository(db_path)
    assert reopened.list_redemptions("U1001")[0]["redemption_id"] == order_id
    assert reopened.list_redemptions("U1001")[0]["status"] == "待发货"
    assert reopened.get_points("U1001") == 700


def test_redeem_mock_requires_owned_address_and_rechecks_points_at_confirm():
    repo, graph = setup()
    with pytest.raises(ValueError, match="收货信息"):
        repo.redeem("U1001", "R001", "no-address", address_id=None)
    assert repo.get_points("U1001") == 1200
    assert repo.list_redemptions("U1001") == []

    own = repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")
    foreign = repo.save_address("U1002", "李四", "13900004321", "上海市大学路8号")
    with pytest.raises(ValueError, match="收货信息"):
        repo.redeem("U1001", "R001", "foreign-address", address_id=foreign["address_id"])
    preview = ask(graph, "兑换 R001", session="stale-points")
    assert preview["result"]["address"]["address_id"] == own["address_id"]
    with repo.connection() as conn:
        conn.execute("UPDATE users SET points=0 WHERE user_id='U1001'")
    failed = ask(graph, "确认兑换", session="stale-points")
    assert failed["result"]["status"] == "redemption_failed"
    assert "积分不足" in failed["result"]["message"]
    assert "积分不足" in failed["messages"][-1].content
    assert repo.list_redemptions("U1001") == []


def test_redeem_preview_can_change_to_another_owned_address_before_final_confirmation():
    repo, graph = setup()
    original = repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")
    other = repo.save_address("U1001", "李四", "13900004321", "上海市大学路8号")
    preview = ask(graph, "兑换 R001", session="change-redeem-address")
    assert preview["result"]["address"]["address_id"] == original["address_id"]
    changed = ask(graph, f"改用收货地址 {other['address_id']}", session="change-redeem-address")
    assert changed["result"]["status"] == "awaiting_redemption_confirmation"
    assert changed["result"]["address"]["recipient"] == "李四"
    assert repo.list_redemptions("U1001") == []
    receipt = ask(graph, "确认兑换", session="change-redeem-address")
    assert receipt["result"]["address"]["address_id"] == other["address_id"]
    assert repo.list_redemptions("U1001")[0]["address_id"] == other["address_id"]


@pytest.mark.parametrize("mobile,needs_full_mobile", [
    ("18800002241", False), ("188****2241", True),
])
def test_pending_redeem_new_shipping_details_do_not_target_old_orders(mobile, needs_full_mobile):
    repo, graph = setup(MisroutedShippingModel())
    old = repo.save_address("U1001", "张三", "19900004475", "深圳市演示路7号楼6楼")
    session = "redeem-new-shipping-" + str(needs_full_mobile)
    first = ask(graph, "兑换 R001", session=session)
    assert first["result"]["status"] == "awaiting_redemption_confirmation"
    assert first["result"]["address"]["address_id"] == old["address_id"]

    changed = ask(graph, f"收货人 王五 电话 {mobile} 地址 深圳市演示路7号楼5楼", session=session)
    assert changed["intent"] == "points"
    assert changed["model_info"]["points"]["action_guardrail"] == (
        "pending_redemption_shipping_details"
    )
    assert "query_redemption_orders" not in [item["tool"] for item in changed["tool_trace"]]
    assert changed["points_state"]["resume_redeem"]["reward_id"] == "R001"
    assert changed["response_info"]["mode"] == "llm"
    assert repo.get_points("U1001") == 1200
    assert repo.list_redemptions("U1001") == []

    if needs_full_mobile:
        assert changed["result"]["status"] == "address_details_required"
        assert changed["result"]["missing"] == ["mobile"]
        changed = ask(graph, "电话 18800002241", session=session)
    assert changed["result"]["status"] == "awaiting_points_confirmation"
    assert changed["result"]["operation"] == "address_add"
    assert changed["result"]["address"]["recipient"] == "王五"
    assert changed["result"]["address"]["detail"] == "深圳市演示路7号楼5楼"
    assert changed["response_info"]["mode"] == "llm"

    saved = ask(graph, "确认保存地址", session=session)
    assert saved["result"]["status"] == "awaiting_redemption_confirmation"
    assert saved["result"]["reward"] == "纪念保温杯"
    assert saved["result"]["address"]["recipient"] == "王五"
    assert saved["result"]["address"]["address_id"] != old["address_id"]
    assert saved["response_info"]["mode"] == "llm"
    assert repo.get_points("U1001") == 1200
    assert repo.list_redemptions("U1001") == []

    receipt = ask(graph, "确认兑换", session=session)
    assert receipt["result"]["status"] == "redeemed"
    assert receipt["result"]["address"]["recipient"] == "王五"
    assert repo.list_redemptions("U1001")[0]["recipient"] == "王五"
    assert repo.get_points("U1001") == 700
    assert next(item for item in repo.list_addresses("U1001")
                if item["address_id"] == old["address_id"])["detail"] == "深圳市演示路7号楼6楼"
    assert receipt["response_info"]["mode"] == "llm"


def test_explicit_old_order_change_is_not_reinterpreted_as_pending_redemption_address():
    router = IntentRouter(llm=MisroutedShippingModel())
    decision, info = router.interpret_points(
        "修改旧订单 demo-order-3，收货人 王五，地址 深圳市演示路7号楼5楼",
        pending={"kind": "redeem", "reward_id": "R001"},
    )
    assert decision.action == "order_address"
    assert info.get("action_guardrail") != "pending_redemption_shipping_details"


def test_multiple_addresses_without_default_require_explicit_selection():
    repo, graph = setup()
    repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")
    other = repo.save_address("U1001", "李四", "13900004321", "上海市大学路8号")
    with repo.connection() as conn:
        conn.execute("UPDATE shipping_addresses SET is_default=0 WHERE user_id='U1001'")
    selection = ask(graph, "兑换 R001", session="choose-redeem-address")
    assert selection["result"]["status"] == "address_selection_required"
    assert selection["active_agent"] == "points"
    assert repo.list_redemptions("U1001") == []
    selected = ask(graph, f"使用收货地址 {other['address_id']}", session="choose-redeem-address")
    assert selected["result"]["status"] == "awaiting_redemption_confirmation"
    assert selected["result"]["address"]["recipient"] == "李四"
    assert repo.list_redemptions("U1001") == []


def test_redeem_rejects_address_removed_after_preview():
    repo, graph = setup()
    address = repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")
    preview = ask(graph, "兑换 R001", session="removed-redeem-address")
    assert preview["result"]["status"] == "awaiting_redemption_confirmation"
    with repo.connection() as conn:
        conn.execute("DELETE FROM shipping_addresses WHERE address_id=?", (address["address_id"],))
    failed = ask(graph, "确认兑换", session="removed-redeem-address")
    assert failed["result"]["status"] == "redemption_failed"
    assert "收货信息" in failed["result"]["message"]
    assert "收货信息" in failed["messages"][-1].content
    assert repo.get_points("U1001") == 1200
    assert repo.list_redemptions("U1001") == []


def test_api_and_gradio_callbacks_expose_points_flow_without_full_mobile(monkeypatch, tmp_path):
    repo, graph = setup()
    monkeypatch.setattr(api_module, "graph", graph)
    response = TestClient(api_module.app).post("/chat", json={
        "user_id": "U1001", "session_id": "points-api", "message": "兑换 R001",
    })
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "redemption_address_required"
    assert response.json()["response_info"]["mode"] == "llm"
    assert all(not item["mutating"] for item in response.json()["tool_trace"])

    chat = ChatApplication(str(tmp_path / "ui-points.db"))
    chat.graph = graph
    chat.repo = repo
    _, history, _, details = chat.respond(
        "新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号",
        [], "U1001", "points-ui",
    )
    assert details["result"]["status"] == "awaiting_points_confirmation"
    assert details["response_info"]["mode"] == "llm"
    assert "13800001234" not in str(details) + str(history)
    assert "138****1234" in str(details)


def test_order_status_changed_after_preview_is_rejected_by_mock_receipt():
    repo, graph = setup()
    preview = ask(graph, "取消订单 demo-order-3", user="U1002", session="stale-order")
    assert preview["result"]["status"] == "awaiting_points_confirmation"
    with repo.connection() as conn:
        conn.execute("UPDATE redemptions SET status='运输中' WHERE redemption_id='demo-order-3'")
    result = ask(graph, "确认取消订单", user="U1002", session="stale-order")
    assert result["result"]["status"] == "cancel_order_failed"
    assert result["result"]["code"] == "INVALID_STATUS"
    assert "运输中" in result["messages"][-1].content
    assert repo.list_redemptions("U1002")[3]["status"] == "运输中"


def test_existing_redemptions_table_gets_additive_order_columns(tmp_path):
    db_path = str(tmp_path / "old-points.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users(user_id TEXT PRIMARY KEY,name TEXT NOT NULL,points INTEGER NOT NULL)")
        conn.execute("INSERT INTO users VALUES('U1001','小明',700)")
        conn.execute("CREATE TABLE rewards(reward_id TEXT PRIMARY KEY,name TEXT NOT NULL,"
                     "points_cost INTEGER NOT NULL,stock INTEGER)")
        conn.execute("INSERT INTO rewards VALUES('R001','纪念保温杯',500,19)")
        conn.execute("CREATE TABLE redemptions(redemption_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,"
                     "reward_id TEXT NOT NULL,points_cost INTEGER NOT NULL,"
                     "created_at TEXT DEFAULT CURRENT_TIMESTAMP,idempotency_key TEXT UNIQUE NOT NULL)")
        conn.execute("INSERT INTO redemptions(redemption_id,user_id,reward_id,points_cost,idempotency_key) "
                     "VALUES('old-order-1','U1001','R001',500,'old-key')")
    repo = BloodRepository(db_path)
    old = repo.list_redemptions("U1001")[0]
    assert old["redemption_id"] == "old-order-1"
    assert old["status"] == "待填写收货信息"
    assert all(key in old for key in ("paid_at", "shipped_at", "delivered_at", "cancelled_at"))
    assert old["paid_at"] is None and old["cancelled_at"] is None
    assert repo.list_addresses("U1001") == []


def test_explicit_order_cancel_is_not_misread_as_aborting_draft():
    class MisreadingModel(UnderstandingModel):
        def invoke(self, messages):
            text = json.loads(messages[-1].content)["message"]
            if "积分子智能体" in messages[0].content and text == "取消订单 demo-order-3":
                return AIMessage(content='{"action":"cancel","order_id":"demo-order-3"}')
            return super().invoke(messages)

    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=MisreadingModel(), response_llm=FactsReplyModel(), risk_llm=False),
        checkpointer=MemorySaver(),
    ))
    state = ask(graph, "取消订单 demo-order-3", user="U1002", session="misread")
    assert state["understanding"]["action"] == "cancel_order"
    assert state["result"]["status"] == "awaiting_points_confirmation"
    assert repo.list_redemptions("U1002")[3]["status"] == "待发货"
