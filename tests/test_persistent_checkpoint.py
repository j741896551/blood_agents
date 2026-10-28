"""文件 SQLite checkpoint 恢复图状态；不靠历史文本重放业务写操作。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from preview_reply import PreviewReplyModel

from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


def _ask(graph, text, *, user_id="U1001", session_id="resume"):
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": session_id}, "metadata": {"entrypoint": "test"}},
    )


def test_appointment_preview_survives_graph_rebuild_without_replaying_submit(tmp_path, monkeypatch):
    # 显式禁用外部模型：这里只测试 LangGraph checkpoint 与本地 Mock。
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False)))
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    business_path = tmp_path / "business.db"
    repo = BloodRepository(str(business_path))
    repo.seed_demo_data()
    first_graph = build_conversational_graph(repo)
    from local_business.api import MockLegacyBloodAPI

    day = MockLegacyBloodAPI(repo).site_slots[0]["slots"][0]["date"]
    preview = _ask(first_graph, f"预约 {day} 上午到市中心献血屋")
    assert preview["result"]["status"] == "awaiting_confirmation"
    assert repo.list_appointments("U1001") == []
    first_graph.close()  # 模拟进程结束；只关闭连接，不删除 checkpoint。

    checkpoint_path = tmp_path / "business.checkpoints.db"
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_mode & 0o077 == 0

    reopened = BloodRepository(str(business_path))
    second_graph = build_conversational_graph(reopened)
    try:
        # 相同会话号但不同入口用户不可拿到 U1001 的待确认预览。
        other_user = _ask(second_graph, "确认预约", user_id="U1002")
        assert other_user.get("result", {}).get("success") is not True
        assert reopened.list_appointments("U1002") == []

        confirmed = _ask(second_graph, "确认预约")
        assert confirmed["result"]["success"] is True
        assert len(reopened.list_appointments("U1001")) == 1
        assert confirmed["result"]["data"]["appointment_id"] == (
            reopened.list_appointments("U1001")[0].appointment_id
        )
    finally:
        second_graph.close()


def test_old_sqlite_saver_appointment_preview_resumes_with_sqlalchemy(tmp_path, monkeypatch):
    """真实旧图留下的待确认状态可被新 saver 续接，不重放预约工具。"""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from blood_agents.orchestration.conversation import ConversationGraph
    from blood_agents.orchestration.graph import build_graph
    from local_business.api import MockLegacyBloodAPI

    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False)))
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "legacy.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    connection = sqlite3.connect(tmp_path / "legacy.checkpoints.db", check_same_thread=False)
    old_graph = ConversationGraph(build_graph(
        repo=repo, router=IntentRouter.from_env(), legacy_api=api,
        checkpointer=SqliteSaver(connection),
    ), conversation_store=repo, checkpoint_connection=connection)
    day = api.site_slots[0]["slots"][0]["date"]
    assert _ask(old_graph, f"预约 {day} 上午到市中心献血屋")["result"]["status"] == (
        "awaiting_confirmation"
    )
    old_graph.close()
    assert repo.list_appointments("U1001") == []

    reopened = BloodRepository(str(path))
    new_graph = build_conversational_graph(reopened)
    try:
        assert _ask(new_graph, "确认预约")["result"]["success"] is True
        assert len(reopened.list_appointments("U1001")) == 1
    finally:
        new_graph.close()


def test_in_memory_business_repo_keeps_in_memory_checkpointer(monkeypatch):
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter()))
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo)
    assert graph.graph.checkpointer.__class__.__name__ == "InMemorySaver"
    graph.close()


def test_checkpoint_path_must_not_alias_business_database(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter()))
    path = tmp_path / "business.db"
    repo = BloodRepository(str(path))
    monkeypatch.setenv("BLOOD_AGENT_CHECKPOINT_DB", str(path))
    with pytest.raises(ValueError, match="separate persistent"):
        build_conversational_graph(repo)


def test_points_preview_survives_restart_and_confirm_writes_once(tmp_path, monkeypatch):
    class PointsModel:
        def invoke(self, messages):
            request = json.loads(messages[-1].content)
            content = request["message"]
            if "主导航" in messages[0].content:
                return AIMessage(content='{"intent":"points"}')
            if content == "确认兑换":
                return AIMessage(content='{"action":"confirm"}')
            return AIMessage(content='{"action":"redeem","reward_id":"R001"}')

    class ReplyModel:
        def invoke(self, messages):
            facts = json.loads(messages[-1].content)
            prefix = ("尚未执行，等待确认：" if facts["result_status"] ==
                      "awaiting_redemption_confirmation" else "本轮查询结果：")
            return AIMessage(content=prefix + "、".join(facts["required_facts"]))

    monkeypatch.setattr(
        IntentRouter, "from_env",
        classmethod(lambda cls: IntentRouter(
            llm=PointsModel(), response_llm=ReplyModel(), risk_llm=False,
        )),
    )
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    business_path = tmp_path / "points.db"
    repo = BloodRepository(str(business_path))
    repo.seed_demo_data()
    repo.save_address("U1001", "张三", "13800001234", "上海市人民路100号")
    first_graph = build_conversational_graph(repo)
    preview = _ask(first_graph, "兑换 R001", session_id="points-resume")
    assert preview["result"]["status"] == "awaiting_redemption_confirmation"
    assert repo.get_points("U1001") == 1200
    first_graph.close()

    reopened = BloodRepository(str(business_path))
    second_graph = build_conversational_graph(reopened)
    try:
        confirmed = _ask(second_graph, "确认兑换", session_id="points-resume")
        assert confirmed["result"]["status"] == "redeemed"
        assert reopened.get_points("U1001") == 700
        assert len(reopened.list_redemptions("U1001")) == 1
        repeated = _ask(second_graph, "确认兑换", session_id="points-resume")
        assert repeated["result"]["status"] == "no_pending_redemption"
        assert reopened.get_points("U1001") == 700
        assert len(reopened.list_redemptions("U1001")) == 1
    finally:
        second_graph.close()


def test_trusted_session_deletion_clears_history_and_checkpoint_without_touching_other_user(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False)))
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "delete-demo.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    graph = build_conversational_graph(repo)
    from local_business.api import MockLegacyBloodAPI

    day = MockLegacyBloodAPI(repo).site_slots[0]["slots"][0]["date"]
    message = f"预约 {day} 上午到市中心献血屋"
    try:
        assert _ask(graph, message, user_id="U1001", session_id="shared")["result"]["status"] == (
            "awaiting_confirmation"
        )
        assert _ask(graph, message, user_id="U1002", session_id="shared")["result"]["status"] == (
            "awaiting_confirmation"
        )
        old_turn = repo.list_conversation_turns("U1001", "shared")[0]
        assert repo.save_conversation_summary(
            user_id="U1001", session_id="shared", title="预约咨询", summary="用户咨询预约。",
            through_turn_id=old_turn["id"], trigger="test",
        )
        assert graph.delete_session("U1001", "shared") == 1
        assert graph.delete_session("U1001", "shared") == 0  # 重复清理安全
        assert repo.list_conversation_turns("U1001", "shared") == []
        assert repo.list_conversation_sessions("U1001") == []
        assert len(repo.list_conversation_turns("U1002", "shared")) == 1
    finally:
        graph.close()

    reopened = BloodRepository(str(path))
    new_graph = build_conversational_graph(reopened)
    try:
        # 旧待确认状态消失；确认文字不能触发已经删除的预约预览写入。
        assert _ask(new_graph, "确认预约", user_id="U1001", session_id="shared")["result"].get(
            "success"
        ) is not True
        assert reopened.list_appointments("U1001") == []
        assert _ask(new_graph, "确认预约", user_id="U1002", session_id="shared")["result"][
            "success"
        ] is True
        assert len(reopened.list_appointments("U1002")) == 1
    finally:
        new_graph.close()


def test_failed_checkpoint_deletion_blocks_resume_until_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter()))
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "checkpoint-failure.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    graph = build_conversational_graph(repo)
    from local_business.api import MockLegacyBloodAPI

    day = MockLegacyBloodAPI(repo).site_slots[0]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到市中心献血屋")
    saver = graph.graph.checkpointer
    real_delete = saver.delete_thread

    def failed_delete(_key):
        raise OSError("synthetic checkpoint failure")

    monkeypatch.setattr(saver, "delete_thread", failed_delete)
    with pytest.raises(OSError, match="synthetic checkpoint failure"):
        graph.delete_session("U1001", "resume")
    assert repo.is_conversation_session_deletion_pending("U1001", "resume")
    with pytest.raises(OSError, match="synthetic checkpoint failure"):
        _ask(graph, "确认预约")
    assert repo.list_appointments("U1001") == []
    monkeypatch.setattr(saver, "delete_thread", real_delete)
    graph.close()

    # 重建图后依然能看见墓碑；新轮次先清理旧 checkpoint，不能执行旧确认。
    reopened = BloodRepository(str(path))
    new_graph = build_conversational_graph(reopened)
    try:
        answer = _ask(new_graph, "确认预约")
        assert answer["result"].get("success") is not True
        assert reopened.list_appointments("U1001") == []
        assert not reopened.is_conversation_session_deletion_pending("U1001", "resume")
        assert [row["user_message"] for row in reopened.list_conversation_turns(
            "U1001", "resume",
        )] == ["确认预约"]
    finally:
        new_graph.close()


def test_failed_history_cleanup_keeps_tombstone_for_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter()))
    monkeypatch.delenv("BLOOD_AGENT_CHECKPOINT_DB", raising=False)
    path = tmp_path / "history-failure.db"
    repo = BloodRepository(str(path))
    repo.seed_demo_data()
    graph = build_conversational_graph(repo)
    from local_business.api import MockLegacyBloodAPI

    day = MockLegacyBloodAPI(repo).site_slots[0]["slots"][0]["date"]
    _ask(graph, f"预约 {day} 上午到市中心献血屋")
    original_finish = repo.finish_conversation_session_deletion

    def failed_finish(_user_id, _session_id):
        raise OSError("synthetic history failure")

    monkeypatch.setattr(repo, "finish_conversation_session_deletion", failed_finish)
    with pytest.raises(OSError, match="synthetic history failure"):
        graph.delete_session("U1001", "resume")
    assert repo.is_conversation_session_deletion_pending("U1001", "resume")
    # 旧行可供恢复删除流程使用，但待清理时不再从历史浏览入口暴露。
    assert repo.list_conversation_turns("U1001", "resume") == []
    assert repo.list_conversation_sessions("U1001") == []
    with repo.connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE user_id='U1001' AND session_id='resume'"
        ).fetchone()[0] == 1
    monkeypatch.setattr(repo, "finish_conversation_session_deletion", original_finish)
    graph.close()

    reopened = BloodRepository(str(path))
    new_graph = build_conversational_graph(reopened)
    try:
        answer = _ask(new_graph, "确认预约")
        assert answer["result"].get("success") is not True
        assert reopened.list_appointments("U1001") == []
        assert not reopened.is_conversation_session_deletion_pending("U1001", "resume")
        assert [row["user_message"] for row in reopened.list_conversation_turns(
            "U1001", "resume",
        )] == ["确认预约"]
    finally:
        new_graph.close()
