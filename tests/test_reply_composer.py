"""验证各领域回复经模型表达，且模型不能改写已确认的业务结果。"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from blood_agents.conversation import ConversationGraph
from blood_agents.graph import build_graph
from blood_agents.model_gateway import ModelGateway, ModelProvider
from blood_agents.reply_composer import (
    UNAVAILABLE_MESSAGE,
    ReplyComposer,
    _required_facts,
    _validate_answer,
    reply_stream_callback,
)
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


def faithful_answer(facts):
    """测试桩只回显结构化事实，不依赖业务节点拼出的底稿。"""
    if facts["intent"] == "general":
        capabilities = facts["result"].get("capabilities") or []
        return "我来为您说明：\n\n" + "\n".join(
            f"- **能力{index}**：{capability}"
            for index, capability in enumerate(capabilities, start=1)
        )
    status = facts["result_status"]
    prefix = "预约成功：" if facts["intent"] == "appointment" and facts["result_success"] is True else ""
    if status == "appointments_found":
        prefix = "已预约："
    elif status == "no_appointments" or (facts["intent"] == "records" and not facts["result"].get("records")):
        prefix = "暂无预约或献血记录："
    elif status in {"awaiting_redemption_confirmation", "awaiting_points_confirmation"}:
        prefix = "尚未执行，等待确认："
    elif status in {"no_orders", "no_addresses", "no_recommendations", "no_pending_redemption"}:
        prefix = "暂无相关数据："
    answer = "我来为您说明：" + prefix + "、".join(facts["required_facts"])
    if facts["intent"] == "faq" and facts["retrieved_context"]:
        # FAQ 来源不再塞进所有 required_facts；测试桩仍需模拟合格的模型引用。
        hit = facts["retrieved_context"][0]
        if hit.get("source_kind") == "synthetic":
            answer += "这是演示资料。"
        answer += f"来源：{hit['source']}"
    return answer


def test_reply_composer_streams_only_plain_general_navigation():
    class StreamingGateway(ModelGateway):
        def __init__(self):
            super().__init__([ModelProvider("fake", object())])
            self.stream_calls = 0
            self.call_calls = 0

        def stream(self, messages, validate, on_chunk, stage="unknown"):
            self.stream_calls += 1
            on_chunk("这是一条普通导航回复。")
            return validate("这是一条普通导航回复。"), {
                "mode": "llm", "provider": "fake", "error_type": None,
            }

        def call(self, messages, validate, stage="unknown"):
            self.call_calls += 1
            answer = "现有资料不足以回答这个知识问题，建议咨询血站医护人员。"
            return validate(answer), {"mode": "llm", "provider": "fake", "error_type": None}

    gateway = StreamingGateway()
    composer = ReplyComposer(gateway)
    chunks = []
    token = reply_stream_callback.set(chunks.append)
    try:
        general = {
            "messages": [HumanMessage(content="请简单介绍一下")],
            "result": {"status": "general", "capabilities": []},
            "memory_event": {"status": "not_applicable"},
        }
        answer, info = composer.compose("general", general)
        assert answer == "这是一条普通导航回复。"
        assert info["mode"] == "llm"
        assert chunks == ["这是一条普通导航回复。"]

        faq = {
            "messages": [HumanMessage(content="这个知识问题怎么处理")],
            "result": {"status": "no_evidence"},
        }
        answer, info = composer.compose("faq", faq)
        assert "资料不足" in answer
        assert info["mode"] == "llm"
        assert gateway.stream_calls == 1
        assert gateway.call_calls == 1
        assert chunks == ["这是一条普通导航回复。"]
    finally:
        reply_stream_callback.reset(token)


class RephraseModel:
    def __init__(self):
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        facts = json.loads(messages[-1].content)
        return AIMessage(content=faithful_answer(facts))


def test_general_greeting_prefers_grouped_markdown_and_retries_once(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "passthrough")

    class StyleModel:
        def __init__(self):
            self.requests = []

        def invoke(self, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return AIMessage(content="你好，我能协助预约献血、查询积分、献血记录和知识。")
            return AIMessage(content=(
                "你好！我可以帮助你：\n\n"
                "- **献血预约**：预约、查询、取消和签到\n"
                "- **积分与礼品**：积分查询及礼品兑换\n"
                "- **献血记录**：查询历史记录\n"
                "- **献血知识**：回答有依据的知识问题\n\n"
                "当前使用本地 Mock/SQLite 演示数据。想先了解哪一项？"
            ))

    model = StyleModel()
    answer, info = ReplyComposer(model).compose("general", {
        "messages": [HumanMessage(content="你好")],
        "result": {"status": "general", "capabilities": [
            "预约献血", "查询积分和兑换礼品", "查询献血记录", "回答献血知识",
        ], "service_facts": {"business_data_source": "local_mock_sqlite"}},
    })
    assert len(model.requests) == 2
    assert "按主题分组" in model.requests[0][0].content
    assert "Markdown 项目符号列表" in json.loads(model.requests[1][-1].content)["revision_feedback"]
    assert answer.count("- **") == 4
    assert info["mode"] == "llm"


def test_general_format_retry_failure_keeps_valid_first_reply(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "passthrough")

    class FailingRewrite:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="你好，我能帮你预约和查询积分。")
            raise RuntimeError("rewrite unavailable")

    model = FailingRewrite()
    answer, info = ReplyComposer(model).compose("general", {
        "messages": [HumanMessage(content="你好")],
        "result": {"status": "general", "capabilities": ["预约献血", "查询积分"]},
    })
    assert model.calls == 2
    assert answer == "你好，我能帮你预约和查询积分。"
    assert info["mode"] == "llm"
    assert "capabilities_not_grouped" in info["quality_warnings"]


def make_graph(reply_model):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(response_llm=reply_model, risk_llm=False, evidence_llm=False),
        legacy_api=api, checkpointer=MemorySaver(),
    ))
    return repo, api, graph


def ask(graph, text, session="response-model"):
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": session}},
    )


def test_all_existing_domains_use_model_reply_node():
    model = RephraseModel()
    _, _, graph = make_graph(model)
    cases = [
        ("你好", "general", "general_reply"),
        ("我要预约", "appointment", "appointment_reply"),
        ("查询积分", "points", "points_reply"),
        ("查询献血记录", "records", "records_agent_reply"),
        ("献血前可以空腹吗", "faq", "faq_agent_reply"),
        ("识别图片", "ocr", "ocr_agent_reply"),
    ]
    for index, (question, intent, node) in enumerate(cases):
        state = ask(graph, question, session=f"reply-{index}")
        assert state["intent"] == intent
        assert state["messages"][-1].content.startswith("我来为您说明：")
        assert state["response_info"]["mode"] == "llm"
        assert node in [entry["node"] for entry in state["node_trace"]]
    assert len(state["messages"]) == 2  # 领域节点不造答复，只由回复模型产生助手消息。
    assert len(model.requests) == len(cases)
    assert "预约子智能体" in model.requests[1][0].content
    assert "积分子智能体" in model.requests[2][0].content
    general_facts = json.loads(model.requests[0][-1].content)
    assert general_facts["result"]["service_facts"]["business_connection_verified"] is False
    assert "不要把未核实的接口配置说成已实际接入或可用" in model.requests[0][0].content


def test_rejected_previous_site_returns_alternative_without_risk_refusal(monkeypatch):
    monkeypatch.delenv("BLOOD_AGENT_RISK_MODE", raising=False)
    class NaturalAlternativeReply(RephraseModel):
        def invoke(self, messages):
            self.requests.append(messages)
            facts = json.loads(messages[-1].content)
            if "其他网点" not in facts["user_message"]:
                return AIMessage(content=faithful_answer(facts))
            option = facts["available_options"][0]
            return AIMessage(content=(
                f"明天（{facts['calendar']['明天']}）没有其他可约网点。"
                f"大学城献血车在 {option['date']} {option['time_slot']} 还有余{option['remaining']}；"
                "这只是可选时段，尚未提交预约。"
            ))

    model = NaturalAlternativeReply()
    _, api, graph = make_graph(model)
    ask(graph, "我想预约明天下午的献血", session="other-site")
    state = ask(graph, "还有其他网点吗？明天去不了市中心献血屋", session="other-site")
    assert state["result"]["status"] == "slot_unavailable"
    assert state["result"]["excluded_sites"] == ["市中心献血屋"]
    assert state["slots"].get("site") is None
    assert state["slots"]["status"] == "collecting"
    facts = json.loads(model.requests[-1][-1].content)
    assert facts["available_options"] == [{
        "site": "大学城献血车", "date": api.site_slots[1]["slots"][0]["date"],
        "time_slot": "上午", "remaining": 3,
    }]
    assert state["response_info"]["mode"] == "llm"
    assert state["response_info"]["risk"] == {"mode": "passthrough", "allow": True}
    assert "大学城献血车" in state["messages"][-1].content


def test_appointment_reply_rejects_wrong_relative_date_even_when_options_are_correct():
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    calendar = {"今天": str(today), "明天": str(today + timedelta(days=1)),
                "后天": str(today + timedelta(days=2))}
    wrong_tomorrow = today + timedelta(days=2)
    wrong_day_after = today + timedelta(days=3)
    answer = (f"放弃明天（{wrong_tomorrow:%m-%d}）的方案；"
              f"后天（{wrong_day_after:%Y-%m-%d}）暂无可预约时段。")
    import pytest

    with pytest.raises(TypeError, match="相对日期错误"):
        _validate_answer(answer, (), "appointment", {"status": "need_more_info"}, calendar)
    corrected = (f"明天（{today + timedelta(days=1):%m-%d}）不去了，"
                 f"后天（{today + timedelta(days=2):%Y-%m-%d}）有哪些网点？")
    assert _validate_answer(corrected, (), "appointment", {}, calendar) == corrected


def test_unconfirmed_booking_rejects_model_claim_of_success():
    class HallucinatingModel:
        def invoke(self, _messages):
            return AIMessage(content="预约成功，您无需再确认。")

    repo, api, graph = make_graph(HallucinatingModel())
    day = api.site_slots[0]["slots"][0]["date"]
    state = ask(graph, f"预约 {day} 上午到市中心献血屋")
    assert state["result"]["status"] == "preview_unavailable"
    assert state["response_info"]["mode"] == "unavailable"
    assert "预约成功" not in state["messages"][-1].content
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0


def test_donor_ack_then_explicit_confirm_uses_actual_submit_receipt():
    model = RephraseModel()
    repo, api, graph = make_graph(model)
    day = api.site_slots[1]["slots"][0]["date"]
    preview = ask(graph, f"预约 {day} 上午到大学城献血车", session="donor-receipt")
    assert preview["result"]["status"] == "awaiting_confirmation"
    assert preview["result"]["donor_preview"]["kind"] == "previous"
    acknowledged = ask(graph, "是", session="donor-receipt")
    assert acknowledged["result"]["donor_confirmed"] is True
    assert repo.list_appointments("U1001") == []

    submitted = ask(graph, "确认预约", session="donor-receipt")
    receipt = submitted["result"]["data"]["appointment_id"]
    assert submitted["result"]["success"] is True
    assert submitted["response_info"]["mode"] == "llm"
    assert receipt in submitted["messages"][-1].content
    assert repo.list_appointments("U1001")[0].appointment_id == receipt


def test_preview_cannot_be_described_as_waiting_for_submit_receipt():
    class PendingReceiptModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            return AIMessage(content=faithful_answer(facts) + "已提交预约，正在等待接口回执。")

    repo, api, graph = make_graph(PendingReceiptModel())
    day = api.site_slots[0]["slots"][0]["date"]
    preview = ask(graph, f"预约 {day} 上午到市中心献血屋", session="pending-receipt")
    assert preview["result"]["status"] == "preview_unavailable"
    assert preview["response_info"]["mode"] == "unavailable"
    assert preview["messages"][-1].content == UNAVAILABLE_MESSAGE
    assert repo.list_appointments("U1001") == []


def test_unconfirmed_cancel_rejects_model_claim_that_record_was_cancelled():
    class FalseCancellationModel(RephraseModel):
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_status"] == "awaiting_action_confirmation":
                return AIMessage(content=(
                    facts["result"]["message"] + "不过这笔预约的状态已取消，不需要再重复取消。"
                ))
            return AIMessage(content=faithful_answer(facts))

    model = FalseCancellationModel()
    repo, api, graph = make_graph(model)
    slot = api.site_slots[0]["slots"][0]
    created = api.submit_appointment({
        "user_id": "U1001", "site": "市中心献血屋", "date": slot["date"],
        "time_slot": slot["time_slot"], "slot_id": slot["slot_id"],
        "donor": {"kind": "self", "name": "小明", "mobile_masked": "138****5678"},
    }, "false-cancel")
    assert created["success"] is True

    state = ask(graph, f"帮我取消 {slot['date']} 上午市中心献血屋的预约", session="false-cancel")
    assert state["result"]["status"] == "preview_unavailable"
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    assert repo.list_appointments("U1001")[0].status == "booked"


def test_unconfirmed_booking_accepts_explicit_not_yet_successful_reply():
    class NotYetBookedModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            date = facts["selected_slots"]["date"]
            return AIMessage(content=(
                f"已记录你的预约意向：{date}（下午）。你上次是在市中心献血屋献血，"
                "这次是否仍选择这个网点？该日期市中心献血屋可预约：上午余5、下午余2。"
                "还需要你确认献血点；目前尚未预约成功。"
            ))

    gateway = ModelGateway([ModelProvider("primary", NotYetBookedModel())])
    repo, api, graph = make_graph(gateway)
    day = api.site_slots[0]["slots"][0]["date"]
    state = ask(graph, f"我想预约{day}下午的献血", session="not-yet-booked")
    assert state["result"]["status"] == "need_more_info"
    assert state["response_info"]["mode"] == "llm"
    assert state["response_info"]["provider"] == "primary"
    assert "目前尚未预约成功" in state["messages"][-1].content
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0


def test_unconfirmed_booking_still_rejects_positive_claim_after_negative_clause():
    class ContradictoryModel:
        def __init__(self, suffix):
            self.suffix = suffix

        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            return AIMessage(content=faithful_answer(facts) + self.suffix)

    for suffix in ("目前尚未预约成功，但预约成功了。", "未确认却已预约成功。"):
        _, api, graph = make_graph(ContradictoryModel(suffix))
        day = api.site_slots[0]["slots"][0]["date"]
        state = ask(graph, f"我想预约{day}下午的献血", session="contradictory-booking")
        assert state["response_info"]["mode"] == "unavailable"
        assert suffix not in state["messages"][-1].content


def test_reply_gateway_failover_does_not_repeat_submission():
    class BrokenModel:
        def invoke(self, _messages):
            raise TimeoutError("primary unavailable")

    backup = RephraseModel()
    gateway = ModelGateway([
        ModelProvider("primary", BrokenModel()),
        ModelProvider("backup", backup),
    ])
    repo, api, graph = make_graph(gateway)
    day = api.site_slots[0]["slots"][0]["date"]
    ask(graph, f"预约 {day} 上午到市中心献血屋", session="failover")
    state = ask(graph, "确认预约", session="failover")
    assert state["result"]["success"] is True
    assert state["response_info"]["provider"] == "backup"
    assert state["messages"][-1].content.startswith("我来为您说明：")
    assert [item["tool"] for item in state["tool_trace"]].count("submit_appointment") == 1
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 1


def test_response_model_failure_preserves_business_result():
    class BrokenModel:
        def invoke(self, _messages):
            raise TimeoutError("response unavailable")

    _, _, graph = make_graph(BrokenModel())
    state = ask(graph, "查询积分", session="reply-error")
    assert state["result"]["points"] == 1200
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE


def test_points_reply_cannot_change_balance_or_reward_catalog():
    class WrongBalanceModel:
        def invoke(self, _messages):
            return AIMessage(content="您现在有 999 积分，可以兑换任何礼品。")

    _, _, graph = make_graph(WrongBalanceModel())
    state = ask(graph, "查询积分", session="wrong-balance")
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE


def test_points_balance_does_not_require_unsolicited_reward_ids():
    result = {"status": "points_found", "success": True, "points": 1200,
              "rewards": [{"reward_id": "R001", "name": "纪念保温杯", "points_cost": 500}]}
    required = _required_facts("points", {"result": result})
    assert required == ("1200",)
    assert _validate_answer("你当前有 1200 积分。", required, "points", result)


def test_points_preview_accepts_natural_paraphrase_but_not_false_success():
    result = {"status": "awaiting_redemption_confirmation", "success": False,
              "confirm_word": "确认兑换", "message": "尚未兑换，等待用户明确确认。",
              "reward_id": "R001", "reward": "纪念保温杯", "points_cost": 500, "points": 1200}
    required = _required_facts("points", {"result": result})
    natural = "你有 1200 积分，R001 纪念保温杯需要 500 积分。目前还没提交，请回复确认兑换。"
    assert _validate_answer(natural, required, "points", result) == natural
    with pytest.raises(TypeError, match="尚未执行"):
        _validate_answer("你有 1200 积分，R001 纪念保温杯需要 500 积分。请回复确认兑换。",
                         required, "points", result)
    with pytest.raises(TypeError, match="声称已兑换"):
        _validate_answer(natural + "兑换成功。", required, "points", result)


def test_cancel_order_reminder_does_not_require_null_reward_or_price():
    # “确认取消”未达到明确确认词时，接口只返回待确认提示；空礼品字段不是事实。
    result = {
        "status": "awaiting_points_confirmation", "success": False,
        "operation": "cancel_order", "confirm_word": "确认取消订单",
        "order_id": "demo-order-3", "reward": None, "points_cost": None,
        "points": None, "message": "尚未执行；请按预览中的确认词回复。",
    }
    required = _required_facts("points", {"result": result})
    assert required == ("确认取消订单",)
    answer = "订单 demo-order-3 尚未取消，请回复‘确认取消订单’后再提交。"
    assert _validate_answer(answer, required, "points", result) == answer


def test_reward_selection_reply_accepts_paraphrase_but_rejects_false_preview():
    result = {"status": "reward_selection_required", "success": False, "points": 1200,
              "message": "请从礼品目录中指定要兑换的礼品编号。"}
    required = _required_facts("points", {"result": result})
    assert required == ("1200",)
    natural = "你有 1200 积分。请告诉我想选择的礼品编号，我再帮你核对。"
    assert _validate_answer(natural, required, "points", result) == natural
    with pytest.raises(TypeError, match="明确选择"):
        _validate_answer("你有 1200 积分，可以继续。", required, "points", result)
    with pytest.raises(TypeError, match="已有兑换预览"):
        _validate_answer("你有 1200 积分，爱心雨伞已进入待确认状态，请回复确认兑换。",
                         required, "points", result)


def test_missing_shipping_reply_requires_address_and_cannot_claim_order_created():
    result = {"status": "redemption_address_required", "success": False,
              "points": 1200, "reward": "纪念保温杯", "points_cost": 500}
    required = _required_facts("points", {"result": result})
    answer = "你有 1200 积分，纪念保温杯需 500 积分。还没下单，请先填写收货地址。"
    assert _validate_answer(answer, required, "points", result) == answer
    with pytest.raises(TypeError, match="误导用户确认下单"):
        _validate_answer(answer + "请回复确认兑换。", required, "points", result)
    with pytest.raises(TypeError, match="声称已下单"):
        _validate_answer("1200 积分的账户已下单纪念保温杯 500 积分，请补收货地址。",
                         required, "points", result)


def test_address_collection_accepts_paraphrase_of_internal_message():
    result = {"status": "address_details_required", "success": False,
              "missing": ["recipient", "mobile", "address_detail"],
              "message": "请补充收货人、手机号和详细地址；尚未保存。"}
    required = _required_facts("points", {"result": result})
    answer = "新增地址需要收货人、手机号和详细地址。收到后会请你核对，目前还没有保存。"
    assert _validate_answer(answer, required, "points", result) == answer


def test_reply_masks_full_example_phone_before_user_sees_it():
    result = {"status": "address_details_required", "success": False,
              "missing": ["mobile"], "message": "请提供手机号。"}
    answer = _validate_answer("请提供手机号，例如 13812340000。", _required_facts(
        "points", {"result": result}), "points", result)
    assert answer == "请提供手机号，例如 138****0000。"


def test_failure_receipt_requires_outcome_and_reason_not_exact_sentence():
    result = {"status": "cancel_order_failed", "success": False,
              "message": "订单当前为运输中，不能取消"}
    required = _required_facts("points", {"result": result})
    assert _validate_answer("订单正在配送，暂时无法取消。", required, "points", result)
    with pytest.raises(TypeError, match="失败结果"):
        _validate_answer("查询与操作预览。", required, "points", result)
    with pytest.raises(TypeError, match="关键原因"):
        _validate_answer("订单无法取消，请稍后再试。", required, "points", result)


def test_points_address_list_requires_contact_details_not_internal_address_id():
    result = {"status": "addresses_found", "success": True, "addresses": [{
        "address_id": "A12345678", "recipient": "张三", "mobile": "138****1234",
        "detail": "上海市人民路100号", "is_default": True,
    }]}
    required = _required_facts("points", {"result": result})
    assert "A12345678" not in required
    assert _validate_answer("已查到收货地址：张三，138****1234，上海市人民路100号。",
                            required, "points", result)


def test_redeemed_receipt_accepts_natural_success_wording_without_losing_facts():
    result = {"status": "redeemed", "success": True, "reward": "纪念保温杯",
              "points_cost": 500, "redemption_id": "b3b794650c58", "order_status": "待发货",
              "address": {"recipient": "张三", "mobile": "138****1234",
                          "detail": "上海市人民路100号"}}
    required = _required_facts("points", {"result": result})
    for wording in ("兑换已成功完成", "兑换已完成", "兑换成功"):
        answer = (f"{wording}。纪念保温杯消耗 500 积分，兑换单号 b3b794650c58，"
                  "状态待发货；收货人张三，138****1234，上海市人民路100号。")
        assert _validate_answer(answer, required, "points", result) == answer
    with pytest.raises(TypeError, match="未在回复中明确说明"):
        _validate_answer("纪念保温杯消耗 500 积分，兑换单号 b3b794650c58，"
                         "状态待发货；收货人张三，138****1234，上海市人民路100号。",
                         required, "points", result)


def test_points_unconfirmed_reply_rejects_natural_completed_claim():
    result = {"status": "redemption_failed", "success": False}
    with pytest.raises(TypeError, match="未收到兑换成功回执"):
        _validate_answer("兑换已完成。", (), "points", result)
    assert _validate_answer("兑换尚未完成。", (), "points", result) == "兑换尚未完成。"


def test_existing_order_address_receipt_does_not_restart_redemption():
    # 这笔订单已经存在；修改收货信息的成功回执不能被解释为尚待首次兑换。
    result = {"status": "order_address_completed", "success": True,
              "operation": "order_address", "data": {
                  "redemption_id": "demo-order-3", "status": "待发货",
                  "recipient": "王五", "address_detail": "北京市学院路20号",
              }}
    required = _required_facts("points", {"result": result})
    assert _validate_answer(
        "订单 demo-order-3 的收货信息已保存，当前状态为待发货；无需再确认兑换。",
        required, "points", result,
    )
    with pytest.raises(TypeError, match="既有订单"):
        _validate_answer(
            "订单 demo-order-3 的地址已保存，还需要你再单独确认一次兑换，"
            "确认后才会正式提交并扣减积分。",
            required, "points", result,
        )


def test_existing_order_address_wrong_draft_is_regenerated(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "passthrough")
    class CorrectOnRetry:
        def __init__(self):
            self.requests = []

        def invoke(self, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return AIMessage(content=(
                    "订单 demo-order-3 的收货信息已保存，之后请确认兑换才会正式扣分。"
                ))
            return AIMessage(content=(
                "订单 demo-order-3 的收货信息已保存，当前状态待发货；无需再确认兑换。"
            ))

    model = CorrectOnRetry()
    result = {"status": "order_address_completed", "success": True,
              "operation": "order_address", "data": {
                  "redemption_id": "demo-order-3", "status": "待发货",
              }}
    answer, info = ReplyComposer(llm=model).compose("points", {
        "result": result, "messages": [HumanMessage(content="确认填写收货信息")],
    })
    assert len(model.requests) == 2
    assert "已有订单修改与新兑换" in json.loads(model.requests[1][-1].content)["revision_feedback"]
    assert "无需再确认兑换" in answer
    assert info["mode"] == "llm"


def test_success_reply_must_keep_api_appointment_id():
    class MissingIdModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_success"] is True:
                return AIMessage(content="预约成功，已经为您安排好。")
            return AIMessage(content=faithful_answer(facts))

    repo, api, graph = make_graph(MissingIdModel())
    day = api.site_slots[0]["slots"][0]["date"]
    ask(graph, f"预约 {day} 上午到市中心献血屋", session="must-keep-id")
    state = ask(graph, "确认预约", session="must-keep-id")
    assert state["result"]["success"] is True
    assert state["response_info"]["mode"] == "unavailable"
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 1


def test_success_reply_accepts_equivalent_success_wording():
    class EquivalentSuccessModel:
        def __init__(self, wording):
            self.wording = wording

        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_success"] is True:
                return AIMessage(content=faithful_answer(facts).replace("预约成功：", self.wording))
            return AIMessage(content=faithful_answer(facts))

    for index, wording in enumerate(("预约已成功提交：", "您的预约已确认成功：", "已确认您的预约：")):
        _, api, graph = make_graph(EquivalentSuccessModel(wording))
        day = api.site_slots[0]["slots"][0]["date"]
        session = f"equivalent-success-{index}"
        ask(graph, f"预约 {day} 上午到市中心献血屋", session=session)
        booked = ask(graph, "确认预约", session=session)
        assert booked["response_info"]["mode"] == "llm"
        assert booked["result"]["data"]["appointment_id"] in booked["messages"][-1].content


def test_success_reply_still_rejects_denied_success():
    class DeniedSuccessModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_success"] is True:
                return AIMessage(content=faithful_answer(facts).replace("预约成功：", "尚未预约成功："))
            return AIMessage(content=faithful_answer(facts))

    _, api, graph = make_graph(DeniedSuccessModel())
    day = api.site_slots[0]["slots"][0]["date"]
    ask(graph, f"预约 {day} 上午到市中心献血屋", session="denied-success")
    booked = ask(graph, "确认预约", session="denied-success")
    assert booked["response_info"]["mode"] == "unavailable"
    assert booked["messages"][-1].content == UNAVAILABLE_MESSAGE


def test_submitted_appointment_query_uses_model_without_new_submission():
    class QueryModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_status"] == "appointments_found":
                item = facts["appointments"][0]
                return AIMessage(content=(
                    f"查询到您有一笔已预约成功的记录：{item['appointment_date']} "
                    f"{item['time_slot']}，{item['site']}，预约号 {item['appointment_id']}。"
                ))
            return AIMessage(content=faithful_answer(facts))

    repo, api, graph = make_graph(QueryModel())
    day = api.site_slots[0]["slots"][0]["date"]
    ask(graph, f"预约 {day} 上午到市中心献血屋", session="model-query")
    booked = ask(graph, "确认预约", session="model-query")
    queried = ask(graph, "查询我的预约信息", session="model-query")
    assert queried["response_info"]["mode"] == "llm"
    assert booked["result"]["data"]["appointment_id"] in queried["messages"][-1].content
    assert [item["tool"] for item in queried["tool_trace"]] == ["query_user_appointments"]
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 1


def test_appointment_query_rejects_fabricated_id_and_new_submission_claim():
    class WrongQueryModel:
        def __init__(self, suffix):
            self.suffix = suffix

        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            if facts["result_status"] == "appointments_found":
                return AIMessage(content=faithful_answer(facts) + self.suffix)
            return AIMessage(content=faithful_answer(facts))

    for index, suffix in enumerate(("另有预约号 abcdef123456。", "刚刚已为您预约成功。")):
        _, api, graph = make_graph(WrongQueryModel(suffix))
        day = api.site_slots[0]["slots"][0]["date"]
        session = f"wrong-query-{index}"
        ask(graph, f"预约 {day} 上午到市中心献血屋", session=session)
        ask(graph, "确认预约", session=session)
        queried = ask(graph, "查询我的预约信息", session=session)
        assert queried["response_info"]["mode"] == "unavailable"
        assert suffix not in queried["messages"][-1].content


def test_faq_without_retrieval_evidence_still_uses_model_with_empty_evidence():
    model = RephraseModel()
    answer, info = ReplyComposer(model).compose(
        "faq", {"retrieved_context": [], "result": {"citations": []}},
    )
    assert info["mode"] == "llm"
    assert answer.startswith("我来为您说明")
    assert model.requests
    assert "暂无可靠依据" in model.requests[0][0].content


def test_business_reply_without_structured_result_never_calls_model():
    model = RephraseModel()
    answer, info = ReplyComposer(model).compose("appointment", {"result": {}})
    assert answer == UNAVAILABLE_MESSAGE
    assert info["error_type"] == "missing_business_data"
    assert model.requests == []


def test_faq_without_evidence_rejects_medical_advice_after_disclaimer():
    class InventingFaqModel:
        def invoke(self, _messages):
            return AIMessage(content="暂无可靠依据，但空腹可以献血，通常没有问题。")

    answer, info = ReplyComposer(InventingFaqModel()).compose(
        "faq", {"retrieved_context": [], "result": {"status": "no_evidence", "citations": []}},
    )
    assert answer == UNAVAILABLE_MESSAGE
    assert info["mode"] == "unavailable"


def test_faq_without_evidence_retries_unsourced_medical_assessment_details():
    class RevisingFaqModel:
        def __init__(self):
            self.requests = []

        def invoke(self, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return AIMessage(content=(
                    "目前没有足以判断您服药期间能否献血的可靠依据。"
                    "通常需要结合具体用药情况、凝血相关指标综合判断，建议咨询血站医护。"
                ))
            return AIMessage(content=(
                "目前没有足以判断您服药期间能否献血的可靠依据。"
                "请向血站医护人员说明您的用药情况，由他们核实。"
            ))

    model = RevisingFaqModel()
    answer, info = ReplyComposer(model).compose("faq", {
        "messages": [HumanMessage(content="我正在服用华法林，今天能献血吗？")],
        "retrieved_context": [],
        "result": {"status": "no_evidence", "citations": []},
    })
    assert info["mode"] == "llm"
    assert len(model.requests) == 2
    assert "凝血相关指标" not in answer
    assert "医学评估" in model.requests[1][-1].content


def test_faq_without_evidence_rejects_repeated_medical_assessment_details():
    class InventingFaqModel:
        def invoke(self, _messages):
            return AIMessage(content=(
                "目前没有足以判断能否献血的可靠依据。"
                "通常需要结合凝血相关指标和用药剂量进行评估，请咨询医护。"
            ))

    answer, info = ReplyComposer(InventingFaqModel()).compose("faq", {
        "retrieved_context": [],
        "result": {"status": "no_evidence", "citations": []},
    })
    assert answer == UNAVAILABLE_MESSAGE
    assert info["mode"] == "unavailable"


def test_multiple_sites_are_passed_as_structured_options_for_side_by_side_reply():
    class TableModel:
        def __init__(self):
            self.facts = None
            self.system = ""

        def invoke(self, messages):
            self.system = messages[0].content
            self.facts = json.loads(messages[-1].content)
            options = self.facts["available_options"]
            sites = list(dict.fromkeys(option["site"] for option in options))
            columns = ["<br>".join(
                f"{option['date']} {option['time_slot']}（余{option['remaining']}）"
                for option in options if option["site"] == site
            ) for site in sites]
            return AIMessage(content=(
                "可以选择这些网点和时间：\n\n| " + " | ".join(sites) + " |\n| "
                + " | ".join("---" for _ in sites) + " |\n| "
                + " | ".join(columns) + " |\n\n您想选哪个网点和时段？"
            ))

    model = TableModel()
    _, api, graph = make_graph(model)
    state = ask(graph, "我要预约", session="site-layout")
    assert state["response_info"]["mode"] == "llm"
    assert "| 市中心献血屋 | 大学城献血车 |" in state["messages"][-1].content
    assert "2026" in state["messages"][-1].content
    assert len(model.facts["available_options"]) == 3
    assert model.facts["result"]["status"] == "need_more_info"
    assert "slot_id" not in json.dumps(model.facts, ensure_ascii=False)
    assert "横向排版" in model.system
    assert api.site_slots[0]["slots"][0]["date"] in state["messages"][-1].content
