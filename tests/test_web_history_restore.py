"""Gradio 历史回显只读测试；使用合成数据，不连接模型平台。"""

from blood_agents.web import NEW_CONVERSATION_WELCOME, ChatApplication, build_ui


def _save_turn(app, trace_id, user_id, session_id, message, answer):
    app.repo.save_conversation_turn(
        trace_id=trace_id, user_id=user_id, session_id=session_id,
        user_message=message, assistant_message=answer,
        intent="faq", entrypoint="gradio",
    )


def test_restore_history_after_app_rebuild_is_read_only_and_isolated(tmp_path):
    db_path = str(tmp_path / "web-history.db")
    first_app = ChatApplication(db_path)
    try:
        _save_turn(first_app, "trace-1", "U1001", "saved-session", "你好", "你好，我可以帮您查询。")
        _save_turn(first_app, "trace-2", "U1001", "saved-session", "还有别的吗", "可以继续提问。")
        _save_turn(first_app, "trace-3", "U1002", "saved-session", "别人提问", "别人回复")
        _save_turn(first_app, "trace-4", "U1001", "other-session", "别的会话", "别的答复")
    finally:
        first_app.graph.close()

    restored_app = ChatApplication(db_path)
    # 加载历史不能调用图，更不能重放旧消息或业务写工具。
    class NoInvoke:
        def invoke(self, *args, **kwargs):
            raise AssertionError("restore must not invoke the graph")

    restored_app.graph.close()
    restored_app.graph = NoInvoke()
    history, details, summary = restored_app.restore_history(" U1001 ", " saved-session ")
    assert history == [
        ("你好", "你好，我可以帮您查询。"),
        ("还有别的吗", "可以继续提问。"),
    ]
    assert details["history_restore"]["status"] == "loaded"
    assert details["history_restore"]["count"] == 2
    assert "未调用模型或业务接口" in summary
    assert restored_app.restore_history("U1002", "other-session")[0] == []
    assert restored_app.restore_history("U1001", "other-session")[0] == [
        ("别的会话", "别的答复"),
    ]


def test_restore_history_masks_sensitive_legacy_rows_and_handles_missing_identity(tmp_path):
    app = ChatApplication(str(tmp_path / "masked.db"))
    try:
        # 模拟过滤规则上线前留下的旧行，确认回显路径会再次过滤。
        with app.repo.connection() as conn:
            conn.execute(
                "INSERT INTO conversation_turns(trace_id,user_id,session_id,user_message,"
                "assistant_message,intent,entrypoint) VALUES(?,?,?,?,?,?,?)",
                ("trace-sensitive", "U1001", "s", "手机号 13812345678", "收到", "faq", "gradio"),
            )
        history, _, _ = app.restore_history("U1001", "s")
        assert "13812345678" not in str(history)
        assert "未保存" in history[0][0]
        assert app.restore_history("", "s")[0] == []
        assert app.restore_history("U1001", "")[1]["history_restore"]["status"] == "missing_identity"
    finally:
        app.graph.close()


def test_restore_history_database_error_is_safe_and_button_is_wired(tmp_path, monkeypatch):
    app = ChatApplication(str(tmp_path / "error.db"))
    try:
        def fail(*args, **kwargs):
            raise RuntimeError("internal database details")

        monkeypatch.setattr(app.repo, "list_conversation_turns", fail)
        history, details, summary = app.restore_history("U1001", "s")
        assert history == []
        assert details["history_restore"] == {"status": "failed", "error_type": "RuntimeError"}
        assert "internal database details" not in summary

        demo = build_ui(app)
        assert demo.get_api_info()
        assert any(getattr(block, "value", None) == "加载此会话历史"
                   for block in demo.blocks.values())
        assert any(getattr(block.fn, "__func__", None) is ChatApplication.restore_with_sidebar
                   for block in demo.fns.values())
        assert sum(getattr(block.fn, "__func__", None) is ChatApplication.new_session
                   for block in demo.fns.values()) == 2  # 左侧按钮与 Chatbot 自带清空图标
    finally:
        app.graph.close()


def test_web_stream_callback_updates_chat_then_replaces_partial_with_final(tmp_path, monkeypatch):
    from blood_agents.llm.reply_composer import reply_stream_callback

    app = ChatApplication(str(tmp_path / "stream-ui.db"))
    try:
        def fake_reply(message, history, user_id, session_id):
            callback = reply_stream_callback.get()
            callback("临时片段")
            final_history = (history or []) + [(message, "智能回复服务暂时不可用，请稍后重试。")]
            return "", final_history, session_id, {"response_info": {"mode": "unavailable"}}, "最终结果"

        monkeypatch.setattr(app, "respond_with_summary", fake_reply)
        updates = list(app.respond_with_stream("你好", [], "U1001", "stream-session"))
        assert len(updates) == 2
        assert updates[0][1][-1] == ("你好", "临时片段")
        # 最终失败结果替换气泡中的局部草稿，而不是继续留下半截回复。
        assert updates[-1][1][-1][1] == "智能回复服务暂时不可用，请稍后重试。"
        assert updates[-1][4] == "最终结果"
    finally:
        app.graph.close()


def test_sidebar_lists_latest_sessions_and_switches_by_user(tmp_path):
    app = ChatApplication(str(tmp_path / "sidebar.db"))
    try:
        _save_turn(app, "t1", "U1001", "session-a", "我想预约献血", "需要先确认时间。")
        _save_turn(app, "t2", "U1001", "session-b", "查询积分", "当前积分为演示数据。")
        _save_turn(app, "t3", "U1001", "session-a", "明天下午", "可继续选择网点。")
        _save_turn(app, "t4", "U1002", "session-c", "查询记录", "这是另一个用户。")

        class NoInvoke:
            def invoke(self, *args, **kwargs):
                raise AssertionError("sidebar callbacks must not invoke the graph")

        app.graph.close()
        app.graph = NoInvoke()
        rows = app.repo.list_conversation_sessions("U1001")
        assert [row["session_id"] for row in rows] == ["session-a", "session-b"]
        assert rows[0]["first_message"] == "我想预约献血"
        assert rows[0]["turn_count"] == 2
        radio, selected, history, _, _, _ = app.load_user_sessions("U1001")
        assert selected == "session-a"
        assert radio["value"] == "session-a"
        assert "待生成标题" in radio["choices"][0][0]
        assert "我想预约献血" not in radio["choices"][0][0]
        assert history == [("我想预约献血", "需要先确认时间。"),
                           ("明天下午", "可继续选择网点。")]

        radio_after_select, selected, history, _, _, _ = app.select_session("U1001", "session-b")
        assert radio_after_select["value"] == "session-b"
        assert selected == "session-b"
        assert history == [("查询积分", "当前积分为演示数据。")]
        assert app.select_session("U1002", "session-a")[2] == []
        other_radio, other_id, other_history, _, _, _ = app.load_user_sessions("U1002")
        assert other_id == "session-c"
        assert other_history == [("查询记录", "这是另一个用户。")]
        assert all(value != "session-a" for _, value in other_radio["choices"])

        new_radio, new_id, new_history, _, _, _ = app.new_session("U1001")
        assert new_id not in {"session-a", "session-b"}
        assert new_radio["choices"][0] == ("新对话（尚未保存）", new_id)
        assert new_history == [(None, NEW_CONVERSATION_WELCOME)]
        saved_radio = app.select_session("U1001", "session-a")[0]
        assert new_id not in {value for _, value in saved_radio["choices"]}
        assert len(app.repo.list_conversation_sessions("U1001")) == 2
        assert app.load_user_sessions("U9999")[2] == [(None, NEW_CONVERSATION_WELCOME)]
    finally:
        # NoInvoke 已替换并关闭原图；其余测试路径也可安全关闭。
        if hasattr(app.graph, "close"):
            app.graph.close()


def test_new_conversation_welcome_is_display_only_and_renderable(tmp_path):
    import gradio as gr
    from langchain_core.messages import AIMessage

    app = ChatApplication(str(tmp_path / "welcome.db"))
    try:
        class NoInvoke:
            def invoke(self, *args, **kwargs):
                raise AssertionError("欢迎引导不应调用业务图或模型")

        app.graph.close()
        app.graph = NoInvoke()
        radio, session_id, history, details, _, _ = app.load_user_sessions("U1001")
        assert radio["value"] == session_id
        assert history == [(None, NEW_CONVERSATION_WELCOME)]
        assert details == {}
        assert "- **献血预约**" in history[0][1]
        assert "- **积分与礼品**" in history[0][1]
        assert "本地 Mock/SQLite 演示数据" in history[0][1]
        # Gradio 的成对消息允许只有助手一侧；页面应能直接显示 Markdown 列表。
        assert gr.Chatbot(type="tuples").postprocess(history).root[0][0] is None
        assert app.repo.list_conversation_sessions("U1001") == []

        new_id = app.new_session("U1001")[1]
        assert new_id != session_id
        assert app.repo.list_conversation_sessions("U1001") == []
        assert app.select_session("U1001", session_id)[2] == []

        class CaptureGraph:
            def invoke(self, payload, config):
                # UI 欢迎语不能变成用户消息、模型上下文或业务操作输入。
                assert [item.content for item in payload["messages"]] == ["查询积分"]
                assert config["configurable"]["thread_id"] == session_id
                return {"messages": [AIMessage(content="积分查询演示答复")],
                        "intent": "points", "result": {"status": "ok"}}

        app.graph = CaptureGraph()
        _, replied, _, _ = app.respond("查询积分", history, "U1001", session_id)
        assert replied == [
            (None, NEW_CONVERSATION_WELCOME),
            ("查询积分", "积分查询演示答复"),
        ]
    finally:
        if hasattr(app.graph, "close"):
            app.graph.close()


def test_first_reply_digest_is_saved_and_sidebar_never_uses_first_message(
    tmp_path, monkeypatch,
):
    app = ChatApplication(str(tmp_path / "titles.db"))
    try:
        with app.repo.connection() as conn:
            conn.execute(
                "INSERT INTO conversation_turns(trace_id,user_id,session_id,user_message,"
                "assistant_message,intent,entrypoint) VALUES(?,?,?,?,?,?,?)",
                ("legacy", "U1001", "legacy-session", "手机号 13812345678", "收到", "faq", "gradio"),
            )
        radio, _, _, _, _, _ = app.load_user_sessions("U1001")
        assert "13812345678" not in str(radio)
        assert "待生成标题" in radio["choices"][0][0]

        from blood_agents.llm.conversation_summarizer import ConversationDigest

        class StubSummarizer:
            def summarize(self, turns):
                assert turns == [app.repo.list_conversation_turns("U1001", new_id)[0]]
                return ConversationDigest(title="积分余额与兑换咨询",
                                          summary="用户询问积分，助手给出当前演示答复。"), {"mode": "llm"}

        app.summarizer = StubSummarizer()

        def fake_reply(message, history, user_id, session_id):
            _save_turn(app, "new-turn", user_id, session_id, message, "演示回复")
            return "", [(message, "演示回复")], session_id, {
                "trace_id": "new-turn", "user_id": user_id, "session_id": session_id,
                "conversation_persistence": {"status": "saved"},
            }, "已完成本轮"

        monkeypatch.setattr(app, "respond_with_summary", fake_reply)
        new_id = app.new_session("U1001")[1]
        result = app.respond_with_sidebar("查询积分", [], "U1001", new_id)
        assert result[5]["value"] == new_id
        assert "积分余额与兑换咨询" in result[5]["choices"][0][0]
        assert "查询积分" not in result[5]["choices"][0][0]
        row = app.repo.list_conversation_sessions("U1001")[0]
        assert row["session_id"] == new_id
        assert row["summary"] == "用户询问积分，助手给出当前演示答复。"
        assert row["summary_turn_id"] == row["last_id"]
        assert "积分余额与兑换咨询" in app.load_user_sessions("U1001")[0]["choices"][0][0]
    finally:
        app.graph.close()


def test_sidebar_titles_distinguish_identical_digest_titles():
    first = {"title": "献血预约咨询", "turn_count": 1, "session_id": "session-000001"}
    second = {"title": "献血预约咨询", "turn_count": 1, "session_id": "session-000002"}
    first_label, first_id = ChatApplication._session_choice(first)
    second_label, second_id = ChatApplication._session_choice(second)
    assert first_label != second_label
    assert first_id != second_id


def test_opening_legacy_session_generates_once_and_model_failure_keeps_history(tmp_path):
    from blood_agents.llm.conversation_summarizer import ConversationDigest, ConversationSummarizer

    app = ChatApplication(str(tmp_path / "legacy-title.db"))
    try:
        _save_turn(app, "legacy-one", "U1001", "old", "我想查积分", "可以帮您查询积分。")

        class StubSummarizer:
            calls = 0

            def summarize(self, turns):
                self.calls += 1
                assert turns[0]["user_message"] == "我想查积分"
                return ConversationDigest(title="个人积分查询",
                                          summary="用户想查询积分，助手表示可以协助查询。"), {}

        stub = StubSummarizer()
        app.summarizer = stub
        result = app.summarize_loaded_session("U1001", "old")
        assert "个人积分查询" in result["choices"][0][0]
        assert stub.calls == 1
        app.summarize_loaded_session("U1001", "old")
        assert stub.calls == 1
        assert app.repo.list_conversation_sessions("U1002") == []

        _save_turn(app, "legacy-two", "U1001", "failed", "查询订单", "服务不可用")
        app.summarizer = ConversationSummarizer(None)
        failed = app.summarize_loaded_session("U1001", "failed")
        assert "待生成标题" in failed["choices"][0][0]
        assert app.restore_history("U1001", "failed")[0] == [("查询订单", "服务不可用")]
    finally:
        app.graph.close()
