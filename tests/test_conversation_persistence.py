"""对话落库是旁路记录，不得改变 LangGraph 的业务执行结果。"""

from langchain_core.messages import AIMessage, HumanMessage

from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.storage.privacy import REDACTED_MESSAGE, sanitize_conversation_text
from local_business.repository import BloodRepository


class StubGraph:
    calls = 0

    def invoke(self, state, config=None):
        self.calls += 1
        return {
            "messages": [*state["messages"], AIMessage(content="已收到，13800005678。")],
            "intent": "faq",
        }


def _invoke(graph, user_id, session_id, message="我的手机号是 13800005678"):
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": session_id}, "metadata": {"entrypoint": "test"}},
    )


def test_turn_persisted_and_isolated_after_database_reopen(tmp_path):
    path = tmp_path / "demo.db"
    repo = BloodRepository(str(path))
    graph = ConversationGraph(StubGraph(), conversation_store=repo)
    first = _invoke(graph, "U1001", "shared")
    _invoke(graph, "U1002", "shared", "你好")
    _invoke(graph, "U1001", "other", "你好")

    assert first["conversation_persistence"] == {"status": "saved"}
    reopened = BloodRepository(str(path))
    assert path.stat().st_mode & 0o077 == 0
    turns = reopened.list_conversation_turns("U1001", "shared")
    assert len(turns) == 1
    assert turns[0]["trace_id"] == first["trace_id"]
    assert turns[0]["entrypoint"] == "test"
    assert turns[0]["user_message"] == REDACTED_MESSAGE
    assert turns[0]["assistant_message"] == REDACTED_MESSAGE
    assert len(reopened.list_conversation_turns("U1002", "shared")) == 1
    assert len(reopened.list_conversation_turns("U1001", "other")) == 1
    assert reopened.list_conversation_turns("U1002", "other") == []


def test_recent_turns_keep_chronological_order_and_validate_limit():
    repo = BloodRepository()
    graph = ConversationGraph(StubGraph(), conversation_store=repo)
    _invoke(graph, "U1001", "session", "第一轮")
    _invoke(graph, "U1001", "session", "第二轮")
    _invoke(graph, "U1001", "session", "第三轮")
    assert [row["user_message"] for row in repo.list_conversation_turns(
        "U1001", "session", limit=2,
    )] == ["第二轮", "第三轮"]
    try:
        repo.list_conversation_turns("U1001", "session", limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError("limit=0 应被拒绝")


def test_persistence_failure_does_not_replay_business_graph():
    class FailedStore:
        def save_conversation_turn(self, **kwargs):
            raise OSError("test database failure")

    business_graph = StubGraph()
    graph = ConversationGraph(business_graph, conversation_store=FailedStore())
    result = _invoke(graph, "U1001", "session")
    assert business_graph.calls == 1
    assert result["messages"][-1].content == "已收到，13800005678。"
    assert result["conversation_persistence"] == {"status": "failed", "error_type": "OSError"}


def test_sensitive_text_is_omitted_at_repository_boundary():
    assert sanitize_conversation_text("我想预约明天下午") == "我想预约明天下午"
    for text in (
        "收件人：张三", "收货地址是北京市朝阳区幸福路8号",
        "北京市朝阳区幸福路8号", "邮箱 a@example.com", "13800005678",
        "身份证号 310101199202021234",
    ):
        assert sanitize_conversation_text(text) == REDACTED_MESSAGE
    repo = BloodRepository()
    repo.save_conversation_turn(
        trace_id="direct", user_id="U1001", session_id="s", user_message="收件人：张三",
        assistant_message="好的，地址是北京市朝阳区幸福路8号", intent="points",
        entrypoint="test",
    )
    row = repo.list_conversation_turns("U1001", "s")[0]
    assert row["user_message"] == REDACTED_MESSAGE
    assert row["assistant_message"] == REDACTED_MESSAGE


def test_retention_is_opt_in_and_only_removes_conversation_rows(tmp_path):
    path = tmp_path / "history.db"
    repo = BloodRepository(str(path), conversation_retention_days=0)
    repo.seed_demo_data()
    repo.save_conversation_turn(
        trace_id="old", user_id="U1001", session_id="s", user_message="旧轮次",
        assistant_message="旧答复", intent="faq", entrypoint="test",
    )
    with repo.connection() as conn:
        conn.execute(
            "UPDATE conversation_turns SET created_at='2000-01-01 00:00:00' WHERE trace_id='old'"
        )
    old_id = repo.list_conversation_turns("U1001", "s")[0]["id"]
    assert repo.save_conversation_summary(
        user_id="U1001", session_id="s", title="旧会话摘要",
        summary="用户进行了旧版演示对话，现在等待过期清理。",
        through_turn_id=old_id, trigger="first_reply",
    )
    repo.save_conversation_turn(
        trace_id="new", user_id="U1002", session_id="s", user_message="新轮次",
        assistant_message="新答复", intent="faq", entrypoint="test",
    )
    assert len(repo.list_conversation_turns("U1001", "s")) == 1  # 默认不删旧数据

    retained = BloodRepository(str(path), conversation_retention_days=30)
    retained.save_conversation_turn(
        trace_id="newer", user_id="U1002", session_id="s", user_message="又一轮",
        assistant_message="答复", intent="faq", entrypoint="test",
    )
    assert retained.list_conversation_turns("U1001", "s") == []
    assert len(retained.list_conversation_turns("U1002", "s")) == 2
    with retained.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_session_summaries").fetchone()[0] == 0
    assert retained.get_points("U1001") == 1200  # 清理不能触碰业务表


def test_explicit_session_and_user_history_deletion_is_scoped():
    repo = BloodRepository()
    for trace, user, session in (
        ("a", "U1001", "s1"), ("b", "U1001", "s2"), ("c", "U1002", "s1"),
    ):
        repo.save_conversation_turn(
            trace_id=trace, user_id=user, session_id=session, user_message="你好",
            assistant_message="你好", intent="faq", entrypoint="test",
        )
    assert repo.delete_conversation_session("U1001", "s1") == 1
    assert repo.list_conversation_turns("U1001", "s1") == []
    assert len(repo.list_conversation_turns("U1001", "s2")) == 1
    assert len(repo.list_conversation_turns("U1002", "s1")) == 1
    assert repo.delete_user_conversation_history("U1001") == 1
    assert repo.list_conversation_turns("U1001", "s2") == []
    assert len(repo.list_conversation_turns("U1002", "s1")) == 1


def test_legacy_history_is_sanitized_on_read_without_rewriting_database():
    repo = BloodRepository()
    with repo.connection() as conn:
        conn.execute(
            "INSERT INTO conversation_turns(trace_id,user_id,session_id,user_message,"
            "assistant_message) VALUES(?,?,?,?,?)",
            ("legacy", "U1001", "s", "收件人：张三", "地址是北京市朝阳区幸福路8号"),
        )
    visible = repo.list_conversation_turns("U1001", "s")[0]
    assert visible["user_message"] == REDACTED_MESSAGE
    assert visible["assistant_message"] == REDACTED_MESSAGE
    with repo.connection() as conn:
        raw = conn.execute("SELECT user_message FROM conversation_turns WHERE trace_id='legacy'").fetchone()
    assert raw[0] == "收件人：张三"  # 旧行未被静默删除或改写


def test_session_summary_is_isolated_versioned_and_deleted_with_history():
    repo = BloodRepository()
    for trace, user in (("u1", "U1001"), ("u2", "U1002")):
        repo.save_conversation_turn(
            trace_id=trace, user_id=user, session_id="shared", user_message="查询积分",
            assistant_message="本轮查询已完成", intent="points", entrypoint="test",
        )
        turn = repo.list_conversation_turns(user, "shared")[0]
        assert repo.save_conversation_summary(
            user_id=user, session_id="shared", title=f"{user}积分咨询",
            summary="用户查询积分，助手完成了本轮答复。",
            through_turn_id=turn["id"], trigger="first_reply",
        )
    assert repo.list_conversation_sessions("U1001")[0]["title"] == "U1001积分咨询"
    assert repo.list_conversation_sessions("U1002")[0]["title"] == "U1002积分咨询"

    old_id = repo.list_conversation_turns("U1001", "shared")[0]["id"]
    repo.save_conversation_turn(
        trace_id="u1-next", user_id="U1001", session_id="shared", user_message="还有什么礼品",
        assistant_message="这里是可兑换礼品", intent="points", entrypoint="test",
    )
    assert not repo.save_conversation_summary(
        user_id="U1001", session_id="shared", title="过时标题",
        summary="这是迟到的模型结果，不能覆盖新轮次。",
        through_turn_id=old_id, trigger="first_reply",
    )
    assert repo.list_conversation_sessions("U1001")[0]["title"] == "U1001积分咨询"
    new_id = repo.list_conversation_turns("U1001", "shared")[-1]["id"]
    assert repo.save_conversation_summary(
        user_id="U1001", session_id="shared", title="积分与礼品咨询",
        summary="用户先查询积分，随后了解了可兑换礼品。",
        through_turn_id=new_id, trigger="manual_refresh",
    )
    assert repo.list_conversation_sessions("U1001")[0]["title"] == "积分与礼品咨询"
    assert repo.delete_conversation_session("U1001", "shared") == 2
    assert repo.list_conversation_sessions("U1001") == []
    assert repo.list_conversation_sessions("U1002")[0]["title"] == "U1002积分咨询"
    assert repo.delete_user_conversation_history("U1002") == 1
    with repo.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversation_session_summaries").fetchone()[0] == 0


def test_session_summary_rejects_sensitive_generated_text():
    repo = BloodRepository()
    repo.save_conversation_turn(
        trace_id="one", user_id="U1001", session_id="s", user_message="你好",
        assistant_message="您好", intent="general", entrypoint="test",
    )
    turn_id = repo.list_conversation_turns("U1001", "s")[0]["id"]
    import pytest

    with pytest.raises(ValueError, match="private data"):
        repo.save_conversation_summary(
            user_id="U1001", session_id="s", title="张三的手机号 13812345678",
            summary="用户和助手进行了问候。", through_turn_id=turn_id,
            trigger="first_reply",
        )
