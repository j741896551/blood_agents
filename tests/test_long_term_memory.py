"""长期偏好来自用户明确原话；模型只能提名，服务端核验来源并支持纠正/撤销。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from blood_agents.llm.reply_composer import _validate_answer
from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.long_term_memory import LongTermMemoryService
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


def _ask(graph, user_id, session_id, text):
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": session_id}, "metadata": {"entrypoint": "test"}},
    )


class PreferenceModel:
    def __init__(self):
        self.extraction_calls = 0

    def invoke(self, messages):
        prompt = messages[0].content
        text = json.loads(messages[-1].content)["message"]
        if "只提取用户本轮明确表达" in prompt:
            self.extraction_calls += 1
            if "哪个" in text:
                return AIMessage(content='{"action":"none","key":null,"value":null,"source_quote":null}')
            if "忘记" in text:
                return AIMessage(content=json.dumps({
                    "action": "revoke", "key": "preferred_site", "value": None,
                    "source_quote": text,
                }, ensure_ascii=False))
            value = "市中心献血屋" if "市中心献血屋" in text else "大学城献血车"
            return AIMessage(content=json.dumps({
                "action": "set", "key": "preferred_site", "value": value,
                "source_quote": text,
            }, ensure_ascii=False))
        if "主导航智能体" in prompt:
            return AIMessage(content='{"intent":"general"}')
        raise AssertionError("unexpected model stage")


class PreferenceReplyModel:
    def __init__(self):
        self.facts = []

    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        self.facts.append(facts)
        event = facts["memory_event"]
        if event["status"] == "saved":
            return AIMessage(content=f"已记住你更偏好的网点：{event['value']}。预约仍需你确认。")
        if event["status"] == "revoked":
            return AIMessage(content="已清除你的网点偏好，后续不会按旧偏好推荐。")
        preferences = facts["long_term_preferences"]
        if preferences:
            return AIMessage(content=f"你此前明确表达的网点偏好是{preferences[0]['value']}，"
                                     "是否可预约仍要看当前网点数据。")
        return AIMessage(content="当前没有保存的网点偏好。")


def test_explicit_preference_cross_session_correction_and_revoke(tmp_path, monkeypatch):
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "preference-demo.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    model = PreferenceModel()
    reply = PreferenceReplyModel()
    graph = build_conversational_graph(repo=repo, router=IntentRouter(
        llm=model, response_llm=reply, risk_llm=False,
    ))
    try:
        saved = _ask(graph, "U1001", "first", "以后我更喜欢大学城献血车，请记住")
        assert saved["memory_event"] == {
            "status": "saved", "key": "preferred_site", "value": "大学城献血车",
        }
        assert saved["intent"] == "general"  # 记录偏好不是提交预约
        assert saved["response_info"]["mode"] == "llm"
        assert repo.list_appointments("U1001") == []
    finally:
        graph.close()

    reopened_repo = BloodRepository(str(path))
    reopened = build_conversational_graph(repo=reopened_repo, router=IntentRouter(
        llm=model, response_llm=reply, risk_llm=False,
    ))
    try:
        remembered = _ask(reopened, "U1001", "second", "我之前偏好哪个网点？")
        assert remembered["memory_event"]["status"] == "not_applicable"
        assert remembered["long_term_preferences"][0]["value"] == "大学城献血车"
        assert "大学城献血车" in remembered["messages"][-1].content
        assert "大学城献血车" in reply.facts[-1]["long_term_preferences"][0]["value"]
        count_before_query = model.extraction_calls
        viewed = _ask(reopened, "U1001", "second", "查看我的网点偏好")
        assert viewed["intent"] == "general"
        assert viewed["long_term_preferences"][0]["value"] == "大学城献血车"
        assert model.extraction_calls == count_before_query

        other = _ask(reopened, "U1002", "second", "我之前偏好哪个网点？")
        assert other["long_term_preferences"] == []
        assert "没有保存" in other["messages"][-1].content

        corrected = _ask(reopened, "U1001", "third", "以后改成市中心献血屋，请记住我的网点偏好")
        assert corrected["memory_event"]["status"] == "saved"
        assert [item["value"] for item in reopened_repo.list_active_preferences("U1001")] == [
            "市中心献血屋",
        ]
        revoked = _ask(reopened, "U1001", "fourth", "请忘记我偏好的献血网点")
        assert revoked["memory_event"]["status"] == "revoked"
        assert reopened_repo.list_active_preferences("U1001") == []
        assert _ask(reopened, "U1001", "fifth", "我之前偏好哪个网点？")[
            "long_term_preferences"
        ] == []
        assert _ask(reopened, "U1001", "second", "我之前偏好哪个网点？")[
            "long_term_preferences"
        ] == []  # 旧会话 checkpoint 中的上一轮快照也被仓储最新事件覆盖
        with reopened_repo.connection() as conn:
            events = conn.execute(
                "SELECT action,value,source_session_id,source_trace_id,source_quote "
                "FROM preference_events WHERE user_id='U1001' ORDER BY id"
            ).fetchall()
        assert [row["action"] for row in events] == ["set", "set", "revoke"]
        assert events[0]["source_session_id"] == "first"
        assert events[0]["source_trace_id"] == saved["trace_id"]
        assert events[0]["source_quote"] == "以后我更喜欢大学城献血车，请记住"
    finally:
        reopened.close()


def test_implicit_selection_does_not_call_preference_model_or_store_memory():
    class NeverCall:
        def invoke(self, _messages):
            raise AssertionError("ordinary appointment must not trigger memory extraction")

    repo = BloodRepository()
    service = LongTermMemoryService(repo, NeverCall())
    assert service.capture(
        user_id="U1001", session_id="s", trace_id="t", text="我要预约大学城献血车明天下午",
    )["status"] == "not_applicable"
    assert service.capture(
        user_id="U1001", session_id="s", trace_id="t-other", text="献血前优先吃什么？",
    )["status"] == "not_applicable"
    assert repo.list_active_preferences("U1001") == []


def test_hallucinated_value_and_sensitive_quote_are_not_saved():
    class HallucinatingModel:
        def invoke(self, _messages):
            return AIMessage(content=json.dumps({
                "action": "set", "key": "preferred_site", "value": "市中心献血屋",
                "source_quote": "以后更喜欢大学城献血车",
            }, ensure_ascii=False))

    repo = BloodRepository()
    service = LongTermMemoryService(repo, HallucinatingModel())
    assert service.capture(
        user_id="U1001", session_id="s", trace_id="t1", text="以后更喜欢大学城献血车",
    )["status"] == "rejected_evidence"
    assert service.capture(
        user_id="U1001", session_id="s", trace_id="t2",
        text="记住我更喜欢大学城献血车，手机号是13800001234",
    )["status"] == "skipped_sensitive"
    assert repo.list_active_preferences("U1001") == []

    class TemporaryModel:
        def invoke(self, _messages):
            return AIMessage(content=json.dumps({
                "action": "set", "key": "preferred_site", "value": "大学城献血车",
                "source_quote": "这次我更喜欢大学城献血车",
            }, ensure_ascii=False))

    assert LongTermMemoryService(repo, TemporaryModel()).capture(
        user_id="U1001", session_id="s", trace_id="t3", text="这次我更喜欢大学城献血车",
    )["status"] == "rejected_evidence"
    assert repo.list_active_preferences("U1001") == []


def test_memory_failure_cannot_be_described_as_saved():
    with pytest.raises(TypeError, match="偏好未保存"):
        _validate_answer(
            "已为您记住网点偏好。", (), "general", {},
            memory_event={"status": "failed"},
        )
    assert _validate_answer(
        "这次未能保存偏好，请稍后重试。", (), "general", {},
        memory_event={"status": "failed"},
    )


def test_time_slot_preference_can_be_saved_and_revoked_with_source():
    class TimeModel:
        def invoke(self, messages):
            text = json.loads(messages[-1].content)["message"]
            return AIMessage(content=json.dumps({
                "action": "revoke" if "忘记" in text else "set",
                "key": "preferred_time_slot", "value": None if "忘记" in text else "上午",
                "source_quote": text,
            }, ensure_ascii=False))

    repo = BloodRepository()
    service = LongTermMemoryService(repo, TimeModel())
    assert service.capture(
        user_id="U1001", session_id="s1", trace_id="t1", text="以后请记住我偏好上午时段",
    )["status"] == "saved"
    assert repo.list_active_preferences("U1001")[0]["value"] == "上午"
    assert service.capture(
        user_id="U1001", session_id="s2", trace_id="t2", text="请忘记我偏好的时间段",
    )["status"] == "revoked"
    assert repo.list_active_preferences("U1001") == []


def test_saved_site_preference_does_not_select_appointment_site():
    repo = BloodRepository()
    repo.seed_demo_data()
    repo.save_preference_event(
        user_id="U1001", key="preferred_site", action="set", value="大学城献血车",
        source_session_id="older", source_trace_id="older-trace",
        source_quote="以后我更喜欢大学城献血车",
    )
    graph = build_conversational_graph(repo=repo, router=IntentRouter())
    try:
        result = _ask(graph, "U1001", "new-session", "我想预约献血")
        assert result["long_term_preferences"][0]["value"] == "大学城献血车"
        assert result["result"]["status"] == "need_more_info"
        assert not result["slots"].get("site")
        assert repo.list_appointments("U1001") == []
    finally:
        graph.close()
