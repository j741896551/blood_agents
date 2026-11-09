"""静态标准问答必须唯一、有效且不占用活动流程；命中时零模型/工具调用。"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.knowledge.static_faq_cache import StaticAnswer, StaticFaqCache
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.web import ChatApplication
from local_business.repository import BloodRepository


class CountingModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content='{"intent":"general"}')


def _ask(graph, text, session="cache"):
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": session}},
    )


def test_unique_approved_hit_skips_every_model_and_business_tool(tmp_path):
    repo = BloodRepository(str(tmp_path / "cache.db"))
    repo.seed_demo_data()
    model = CountingModel()
    router = IntentRouter(llm=model, response_llm=model, tool_llm=model)
    graph = ConversationGraph(build_graph(
        repo, router, checkpointer=MemorySaver(), static_cache=StaticFaqCache.from_file(),
    ), conversation_store=repo)

    hit = _ask(graph, "  现在可以上传图片识别吗？  ")
    assert model.calls == 0
    assert hit["response_info"] == {"mode": "static_cache"}
    assert hit["cache_info"] == {
        "status": "hit", "key": "ocr_upload_scope", "version": "2026-10-21.1",
    }
    assert [item["node"] for item in hit["node_trace"]] == ["static_faq_cache"]
    assert hit["tool_trace"] == []
    assert "暂不支持直接上传图片识别" in hit["messages"][-1].content
    assert repo.list_conversation_turns("U1001", "cache")[0]["assistant_message"] == hit["messages"][-1].content

    # 多加一个业务请求，必须走原有主导航，不能因包含相同短语而偷用缓存。
    miss = _ask(graph, "现在可以上传图片识别吗？顺便查询我的积分")
    assert miss["cache_info"] == {"status": "miss", "reason": "no_unique_match"}
    assert model.calls > 0
    assert miss["response_info"]["mode"] != "static_cache"


def test_active_business_flow_does_not_hit_static_cache(tmp_path):
    repo = BloodRepository(str(tmp_path / "active.db"))
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(), checkpointer=MemorySaver(),
        static_cache=StaticFaqCache.from_file(),
    ))
    first = _ask(graph, "我要预约献血", session="active")
    assert first["active_agent"] == "appointment"
    second = _ask(graph, "现在可以上传图片识别吗", session="active")
    assert second["cache_info"] == {"status": "miss", "reason": "active_task"}
    assert second["response_info"]["mode"] != "static_cache"


def test_ambiguous_expired_unapproved_or_corrupt_catalog_falls_back(tmp_path):
    entry = StaticAnswer(
        key="one", version="1", expires_on=date(2027, 1, 1), answer="固定答复",
        questions=("这是演示数据吗",), approved=True, review_basis="user_confirmed",
    )
    assert StaticFaqCache((entry, entry)).lookup("这是演示数据吗", today=date(2026, 10, 21))[1] == "ambiguous"
    assert StaticFaqCache((entry,)).lookup("这是演示数据吗", today=date(2027, 1, 1))[1] == "no_unique_match"
    unapproved = StaticAnswer(**{**entry.__dict__, "approved": False})
    assert StaticFaqCache((unapproved,)).lookup("这是演示数据吗", today=date(2026, 10, 21))[1] == "no_unique_match"
    assert StaticFaqCache((entry,)).lookup("这是演示数据吗", active_agent="points")[1] == "active_task"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    assert StaticFaqCache.from_file(corrupt).lookup("这是演示数据吗")[1] == "catalog_unavailable"


def test_full_question_matching_rejects_personal_or_realtime_variations():
    cache = StaticFaqCache.from_file()
    for text in (
        "现在可以上传图片识别吗？我的积分是多少",
        "现在可以上传图片识别吗，帮我预约",
        "可以通过图片查询我的预约数据吗",
        "现在大学城献血车还有名额吗",
    ):
        entry, reason = cache.lookup(text, today=date(2026, 10, 21))
        assert entry is None and reason == "no_unique_match"


def test_cache_only_gradio_title_does_not_call_model_even_after_reopen(tmp_path, monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(
        lambda cls, registry=None: IntentRouter(llm=model, response_llm=model, registry=registry),
    ))
    db_path = str(tmp_path / "web-cache.db")
    app = ChatApplication(db_path)

    class NoDigest:
        def summarize(self, _turns):
            raise AssertionError("pure cache session must not request a model title")

    app.summarizer = NoDigest()
    try:
        session_id = app.new_session("U1001")[1]
        result = app.respond_with_sidebar("现在可以上传图片识别吗", [], "U1001", session_id)
        assert model.calls == 0
        assert "演示系统标准问答" in result[5]["choices"][0][0]
        row = app.repo.list_conversation_sessions("U1001")[0]
        assert row["summary_trigger"] == "static_cache"
        assert app.repo.list_conversation_turns("U1001", session_id)[0]["response_mode"] == "static_cache"
    finally:
        app.graph.close()

    reopened = ChatApplication(db_path)
    reopened.summarizer = NoDigest()
    try:
        reopened.summarize_loaded_session("U1001", session_id)
        assert model.calls == 0
        assert reopened.repo.list_conversation_sessions("U1001")[0]["summary_trigger"] == "static_cache"

        # 一旦同会话出现真实的非缓存轮次，固定标题可被正常摘要替换。
        from blood_agents.llm.conversation_summarizer import ConversationDigest

        class MixedDigest:
            def summarize(self, _turns):
                return ConversationDigest(
                    title="演示系统与积分咨询", summary="用户先了解演示范围，随后咨询积分相关问题。",
                ), {"mode": "llm"}

        reopened.repo.save_conversation_turn(
            trace_id="later", user_id="U1001", session_id=session_id,
            user_message="查询积分", assistant_message="这是一条合成测试答复。",
            intent="points", entrypoint="test", response_mode="llm",
        )
        reopened.summarizer = MixedDigest()
        reopened.summarize_loaded_session("U1001", session_id)
        row = reopened.repo.list_conversation_sessions("U1001")[0]
        assert row["title"] == "演示系统与积分咨询"
        assert row["summary_trigger"] == "legacy_open"
    finally:
        reopened.graph.close()
