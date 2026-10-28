"""积分子图的多轮确认与事实边界：可控模型只模拟语义，不模拟业务接口。"""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


class PointsUnderstandingModel:
    def invoke(self, messages):
        request = json.loads(messages[-1].content)
        message = request["message"]
        if "主导航" in messages[0].content:
            return AIMessage(content='{"intent":"points"}')
        if message.strip("。！？") == "确认兑换":
            return AIMessage(content='{"action":"confirm"}')
        if "不兑了" in message:
            return AIMessage(content='{"action":"cancel"}')
        if "兑换" in message:
            reward_id = "R001" if "R001" in message else None
            return AIMessage(content=json.dumps({"action": "redeem", "reward_id": reward_id}))
        return AIMessage(content='{"action":"query"}')


class FactsReplyModel:
    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        status = facts["result_status"]
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
        } else "暂无相关数据：" if status in {
            "no_orders", "no_addresses", "no_recommendations", "no_pending_redemption",
        } else "本轮查询结果："
        return AIMessage(content=prefix + "、".join(facts["required_facts"]))


def setup_points_graph(reply_model=None, understanding_model=None, api_factory=MockLegacyBloodAPI):
    repo = BloodRepository()
    repo.seed_demo_data()
    router = IntentRouter(
        llm=understanding_model or PointsUnderstandingModel(),
        response_llm=reply_model or FactsReplyModel(), risk_llm=False,
    )
    graph = ConversationGraph(build_graph(
        repo, router, legacy_api=api_factory(repo), checkpointer=MemorySaver(),
    ))
    return repo, graph


def ask(graph, message, *, user="U1001", session="points-test"):
    return graph.invoke(
        {"user_id": user, "messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": session}},
    )


def seed_address(repo):
    return repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")


def test_address_save_receipt_survives_followup_catalog_query_failure():
    """地址已写入后 MCP 查询失败，不再留下可重复执行的地址确认态。"""
    from blood_agents.domains.points import points_agent

    repo = BloodRepository(":memory:")
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)

    class Tools:
        def invoke(self, name, **kwargs):
            if name == "save_shipping_address":
                return api.save_shipping_address(**kwargs)
            if name == "query_points":
                raise RuntimeError("synthetic read failure after save")
            raise AssertionError(name)

    state = {
        "user_id": "U1001",
        "messages": [HumanMessage(content="确认保存地址")],
        "understanding": {"action": "confirm"},
        "points_state": {
            "kind": "address_add", "recipient": "张三", "mobile": "13800001234",
            "address_detail": "上海市人民路100号",
            "resume_redeem": {"reward_id": "R001"},
        },
        "model_info": {"navigation": {"mode": "llm"}, "points": {"mode": "llm"}},
    }
    result = points_agent(repo, api, Tools())(state)
    assert result["result"]["status"] == "address_add_completed"
    assert result["result"]["success"] is True
    assert result["result"]["resume_redemption_unavailable"] is True
    assert result["points_state"] == {}
    assert len(repo.list_addresses("U1001")) == 1


def test_redeem_requires_delivered_preview_and_explicit_confirmation():
    repo, graph = setup_points_graph()
    address = seed_address(repo)
    preview = ask(graph, "兑换 R001")
    assert preview["result"]["status"] == "awaiting_redemption_confirmation"
    assert preview["points_state"]["reward_id"] == "R001"
    assert preview["result"]["address"]["address_id"] == address["address_id"]
    assert preview["response_info"]["mode"] == "llm"
    assert preview["active_agent"] == "points"
    assert repo.get_points("U1001") == 1200
    assert [item["tool"] for item in preview["tool_trace"]] == [
        "query_points", "list_rewards", "query_shipping_addresses",
    ]

    confirmed = ask(graph, "确认兑换")
    assert confirmed["result"]["status"] == "redeemed"
    assert confirmed["result"]["success"] is True
    assert confirmed["result"]["duplicate"] is False
    assert repo.get_points("U1001") == 700
    assert [item["tool"] for item in confirmed["tool_trace"]] == ["redeem_reward"]
    assert confirmed["points_state"] == {}
    assert confirmed["active_agent"] is None

    repeated = ask(graph, "确认兑换")
    assert repeated["result"]["status"] == "no_pending_redemption"
    assert repo.get_points("U1001") == 700
    assert all(item["tool"] != "redeem_reward" for item in repeated["tool_trace"])


def test_reward_name_from_current_catalog_creates_preview_then_confirms():
    class NameUnderstanding(PointsUnderstandingModel):
        def invoke(self, messages):
            request = json.loads(messages[-1].content)
            if request["message"] == "爱心雨伞" and "主导航" not in messages[0].content:
                return AIMessage(content='{"action":"redeem","reward_id":null}')
            return super().invoke(messages)

    repo, graph = setup_points_graph(understanding_model=NameUnderstanding())
    seed_address(repo)
    preview = ask(graph, "爱心雨伞", session="redeem-by-name")
    assert preview["result"]["status"] == "awaiting_redemption_confirmation"
    assert preview["result"]["reward_id"] == "R002"
    assert preview["points_state"]["reward_id"] == "R002"
    assert preview["response_info"]["mode"] == "llm"
    assert [item["tool"] for item in preview["tool_trace"]] == [
        "query_points", "list_rewards", "query_shipping_addresses",
    ]
    assert repo.get_points("U1001") == 1200

    confirmed = ask(graph, "确认兑换", session="redeem-by-name")
    assert confirmed["result"]["status"] == "redeemed"
    assert confirmed["result"]["reward_id"] == "R002"
    assert [item["tool"] for item in confirmed["tool_trace"]] == ["redeem_reward"]
    assert repo.get_points("U1001") == 400


def test_multiple_reward_names_or_conflicting_id_never_creates_preview():
    repo, graph = setup_points_graph()
    for index, message in enumerate(("兑换爱心雨伞或纪念保温杯", "兑换 R001 爱心雨伞")):
        state = ask(graph, message, session=f"ambiguous-reward-{index}")
        assert state["result"]["status"] == "reward_selection_required"
        assert state["points_state"] == {}
        assert all(item["tool"] != "redeem_reward" for item in state["tool_trace"])
    assert repo.get_points("U1001") == 1200


def test_cancel_preview_does_not_write_and_other_user_cannot_confirm():
    repo, graph = setup_points_graph()
    seed_address(repo)
    ask(graph, "兑换 R001", session="shared")
    other = ask(graph, "确认兑换", user="U1002", session="shared")
    assert other["result"]["status"] == "no_pending_redemption"
    cancelled = ask(graph, "不兑了", session="shared")
    assert cancelled["result"]["status"] == "redemption_aborted"
    assert cancelled["active_agent"] is None
    assert repo.get_points("U1001") == 1200
    assert all(item["tool"] != "redeem_reward" for item in cancelled["tool_trace"])


def test_reply_failure_discards_pending_write_state():
    class BadReply:
        def invoke(self, _messages):
            return AIMessage(content="已经兑换成功")

    repo, graph = setup_points_graph(reply_model=BadReply())
    seed_address(repo)
    preview = ask(graph, "兑换 R001")
    assert preview["response_info"]["mode"] == "unavailable"
    assert preview["points_state"] == {}
    assert preview["active_agent"] is None
    assert repo.get_points("U1001") == 1200
    confirm = ask(graph, "确认兑换")
    assert confirm["result"]["status"] == "no_pending_redemption"
    assert repo.get_points("U1001") == 1200


def test_understanding_failure_never_submits_pending_redemption():
    class FlakyUnderstanding(PointsUnderstandingModel):
        def invoke(self, messages):
            request = json.loads(messages[-1].content)
            if request["message"] == "确认兑换":
                raise TimeoutError("model timeout")
            return super().invoke(messages)

    repo, graph = setup_points_graph(understanding_model=FlakyUnderstanding())
    seed_address(repo)
    ask(graph, "兑换 R001")
    result = ask(graph, "确认兑换")
    assert result["model_info"]["mode"] == "rules_fallback"
    assert result["result"]["status"] == "awaiting_redemption_confirmation"
    assert repo.get_points("U1001") == 1200
    assert all(item["tool"] != "redeem_reward" for item in result["tool_trace"])


def test_points_query_failure_cannot_be_described_as_known_balance():
    class FailedPointsAPI(MockLegacyBloodAPI):
        def get_points(self, user_id):
            return {"success": False, "code": "UPSTREAM_DOWN", "message": "积分接口暂不可用"}

    class InventedBalanceReply:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            return AIMessage(content=f"{facts['result']['message']}，您有1200积分。")

    repo, graph = setup_points_graph(
        reply_model=InventedBalanceReply(), api_factory=FailedPointsAPI,
    )
    state = ask(graph, "查询我的积分")
    assert state["result"]["status"] == "points_query_failed"
    assert state["response_info"]["mode"] == "unavailable"
    assert "1200" not in state["messages"][-1].content
    assert [item["tool"] for item in state["tool_trace"]] == ["query_points"]
    assert repo.get_points("U1001") == 1200


def test_reward_not_in_catalog_never_creates_pending_or_writes():
    repo, graph = setup_points_graph()
    state = ask(graph, "兑换 R999")
    assert state["result"]["status"] == "reward_not_found"
    assert state["points_state"] == {}
    assert state["active_agent"] is None
    assert repo.get_points("U1001") == 1200
    assert all(item["tool"] != "redeem_reward" for item in state["tool_trace"])
