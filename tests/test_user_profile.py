"""用户兴趣画像的增量提取、来源证据、隔离与提示词注入测试。"""

import json
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage

from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.orchestration.user_profile import (
    PROFILE_PROMPT,
    PROFILE_PROMPT_VERSION,
    ProfileExtraction,
    UserProfileSummarizer,
)
from blood_agents.storage.privacy import REDACTED_MESSAGE
from local_business.repository import BloodRepository


def _save_turn(repo, trace_id, user_id, session_id, text, *, response_mode="llm"):
    repo.save_conversation_turn(
        trace_id=trace_id, user_id=user_id, session_id=session_id,
        user_message=text, assistant_message="助手回复文本不得成为画像来源。",
        intent="general", entrypoint="test", response_mode=response_mode,
    )
    return repo.list_conversation_turns(user_id, session_id)[-1]["id"]


class ProfileModel:
    def __init__(self, build_changes):
        self.build_changes = build_changes
        self.inputs = []

    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        self.inputs.append(payload)
        changes = self.build_changes(payload["user_turns"])
        return AIMessage(content=ProfileExtraction(changes=changes).model_dump_json())


def _set_change(turn, value):
    return {
        "key": "hobby", "action": "set", "value": value,
        "source_turn_id": turn["source_turn_id"],
        "source_quote": turn["user_message"],
    }


def test_profile_extracts_stable_hobbies_without_memory_phrase_and_filters_other_data():
    repo = BloodRepository()
    stable_id = _save_turn(repo, "t-stable", "U1001", "session-a", "我平时喜欢徒步，也爱摄影")
    _save_turn(repo, "t-identity", "U1001", "session-b", "我是女性，血型A型，平时喜欢游泳")
    _save_turn(repo, "t-name", "U1001", "session-b", "我叫张三，平时喜欢跑步")
    _save_turn(repo, "t-business", "U1001", "session-b", "查一下献血记录，我喜欢阅读")
    _save_turn(repo, "t-cache", "U1001", "session-c", "我平时喜欢做饭", response_mode="static_cache")
    _save_turn(repo, "t-sensitive", "U1001", "session-c", "我平时喜欢跑步，电话13812345678")

    def changes(turns):
        assert [turn["source_turn_id"] for turn in turns] == [stable_id]
        turn = turns[0]
        return [_set_change(turn, "徒步"), _set_change(turn, "摄影")]

    model = ProfileModel(changes)
    service = UserProfileSummarizer(repo, model)
    outcome = service.refresh("U1001")

    assert outcome["status"] == "updated"
    assert repo.get_user_profile("U1001") == {"hobbies": ["徒步", "摄影"]}
    assert repo.get_user_profile("U1002") == {"hobbies": []}
    serialized_input = json.dumps(model.inputs[0], ensure_ascii=False)
    assert "助手回复文本" not in serialized_input
    assert "女性" not in serialized_input
    assert "血型" not in serialized_input
    assert "献血记录" not in serialized_input
    assert "13812345678" not in serialized_input
    assert "source_quote" not in str(repo.list_user_profile_events("U1001"))
    assert [event["source_turn_id"] for event in repo.list_user_profile_events("U1001")] == [
        stable_id, stable_id,
    ]
    assert all(event["extractor_version"] == PROFILE_PROMPT_VERSION
               for event in repo.list_user_profile_events("U1001"))
    assert repo.get_user_profile_cursor("U1001") > stable_id


def test_one_time_hobby_is_rejected_and_cursor_does_not_reprocess_it():
    repo = BloodRepository()
    turn_id = _save_turn(repo, "t-temp", "U1001", "s", "这次我喜欢爬山")
    model = ProfileModel(lambda turns: [_set_change(turns[0], "爬山")])

    outcome = UserProfileSummarizer(repo, model).refresh("U1001")

    assert outcome["status"] == "no_profile_change"
    assert outcome["rejected_count"] == 1
    assert repo.get_user_profile("U1001") == {"hobbies": []}
    assert repo.get_user_profile_cursor("U1001") >= turn_id
    assert UserProfileSummarizer(repo, model).refresh("U1001")["status"] == "up_to_date"
    assert len(model.inputs) == 1


def test_negative_interest_is_not_saved_as_a_positive_hobby():
    repo = BloodRepository()
    _save_turn(repo, "t-negative", "U1001", "s", "我不喜欢徒步")
    model = ProfileModel(lambda turns: [_set_change(turns[0], "徒步")])

    outcome = UserProfileSummarizer(repo, model).refresh("U1001")

    assert outcome["status"] == "no_profile_change"
    assert outcome["rejected_count"] == 1
    assert repo.get_user_profile("U1001") == {"hobbies": []}


def test_user_can_revoke_and_clear_saved_hobby_with_source_events():
    repo = BloodRepository()
    first_id = _save_turn(repo, "t-set", "U1001", "s1", "我长期喜欢徒步")

    class SequentialModel:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            turn = payload["user_turns"][0]
            text = turn["user_message"]
            if "不再" in text:
                changes = [{
                    "key": "hobby", "action": "revoke", "value": "徒步",
                    "source_turn_id": turn["source_turn_id"], "source_quote": text,
                }]
            elif "忘记" in text:
                changes = [{
                    "key": "hobby", "action": "clear", "value": None,
                    "source_turn_id": turn["source_turn_id"], "source_quote": text,
                }]
            else:
                changes = [_set_change(turn, "徒步")]
            return AIMessage(content=ProfileExtraction(changes=changes).model_dump_json())

    service = UserProfileSummarizer(repo, SequentialModel())
    assert service.refresh("U1001")["status"] == "updated"
    assert repo.get_user_profile("U1001")["hobbies"] == ["徒步"]

    revoke_id = _save_turn(repo, "t-revoke", "U1001", "s2", "我不再喜欢徒步了")
    assert service.refresh("U1001")["status"] == "updated"
    assert repo.get_user_profile("U1001")["hobbies"] == []
    assert repo.list_user_profile_events("U1001")[-1]["source_turn_id"] == revoke_id

    clear_id = _save_turn(repo, "t-clear", "U1001", "s3", "请忘记我所有的爱好")
    assert service.refresh("U1001")["status"] == "updated"
    assert repo.get_user_profile("U1001")["hobbies"] == []
    assert repo.list_user_profile_events("U1001")[-1]["source_turn_id"] == clear_id
    assert repo.list_user_profile_events("U1001")[0]["source_turn_id"] == first_id


def test_model_failure_keeps_profile_cursor_and_repository_enforces_user_source():
    repo = BloodRepository()
    turn_id = _save_turn(repo, "t-fail", "U1001", "s1", "我平时喜欢阅读")

    class FailingModel:
        def invoke(self, _messages):
            raise TimeoutError("synthetic timeout")

    outcome = UserProfileSummarizer(repo, FailingModel()).refresh("U1001")
    assert outcome["status"] == "failed"
    assert repo.get_user_profile_cursor("U1001") == 0

    with repo.connection() as conn:
        source = conn.execute(
            "SELECT id FROM conversation_turns WHERE user_id='U1001' AND id=?", (turn_id,),
        ).fetchone()
    assert source is not None
    try:
        repo.save_user_profile_update(
            user_id="U1002", through_turn_id=turn_id, profile={"hobbies": ["阅读"]},
            changes=[{
                "key": "hobby", "action": "set", "value": "阅读",
                "source_session_id": "s1", "source_turn_id": turn_id,
            }],
        )
    except ValueError as exc:
        assert "source turn" in str(exc)
    else:
        raise AssertionError("another user's conversation turn must not become a profile source")


def test_profile_query_is_routed_to_general_and_only_relevant_prompt_gets_profile():
    repo = BloodRepository()
    source_id = _save_turn(repo, "t-profile", "U1001", "source", "我平时喜欢徒步")
    assert repo.save_user_profile_update(
        user_id="U1001", through_turn_id=source_id, profile={"hobbies": ["徒步"]},
        changes=[{
            "key": "hobby", "action": "set", "value": "徒步",
            "source_session_id": "source", "source_turn_id": source_id,
        }],
    )

    class NavigationModel:
        def invoke(self, _messages):
            # 故意模拟主导航误判为 records，画像意图门禁仍应回到 general。
            return AIMessage(content='{"intent":"records"}')

    class ReplyModel:
        def __init__(self):
            self.facts = []

        def invoke(self, messages):
            self.facts.append(json.loads(messages[-1].content))
            return AIMessage(content="你之前提到喜欢徒步。")

    reply_model = ReplyModel()
    graph = build_conversational_graph(
        repo=repo,
        router=IntentRouter(llm=NavigationModel(), response_llm=reply_model, risk_llm=False),
    )
    try:
        result = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="我有哪些爱好？")]},
            config={"configurable": {"thread_id": "profile-query"}},
        )
        assert result["intent"] == "general"
        assert reply_model.facts[-1]["user_profile"] == {"hobbies": ["徒步"]}

        graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="你好")]},
            config={"configurable": {"thread_id": "unrelated-general"}},
        )
        assert reply_model.facts[-1]["user_profile"] is None

        _save_turn(repo, "t-pending", "U1002", "pending-source", "我平时喜欢篮球")
        graph.invoke(
            {"user_id": "U1002", "messages": [HumanMessage(content="我有哪些爱好？")]},
            config={"configurable": {"thread_id": "pending-query"}},
        )
        assert reply_model.facts[-1]["profile_status"] == "pending_update"
        assert "gender" not in ProfileExtraction.model_json_schema()["$defs"]["ProfileChange"]["properties"]
        assert "blood_type" not in PROFILE_PROMPT
    finally:
        graph.close()


def test_incremental_profile_turns_are_scoped_to_user_and_skip_redacted_messages():
    repo = BloodRepository()
    _save_turn(repo, "t1", "U1001", "s1", "我平时喜欢读书")
    _save_turn(repo, "t2", "U1002", "s2", "我平时喜欢篮球")
    _save_turn(repo, "t3", "U1001", "s3", "我的电话是13812345678")
    turns = repo.list_user_conversation_turns_after("U1001", 0)
    assert all(turn["user_message"] != "我平时喜欢篮球" for turn in turns)
    assert any(turn["user_message"] == REDACTED_MESSAGE for turn in turns)


def test_api_schedules_profile_update_after_persisted_chat_response(monkeypatch):
    from fastapi.testclient import TestClient

    from blood_agents import api

    calls = []

    class GraphWithConversationStore:
        conversation_store = api.repo

        def invoke(self, *_args, **_kwargs):
            return {
                "messages": [AIMessage(content="已收到。")], "intent": "general",
                "trace_id": "trace-profile-api", "result": {}, "model_info": {},
                "response_info": {}, "tool_plan_info": {}, "tool_trace": [],
                "conversation_persistence": {"status": "saved"},
            }

    monkeypatch.setattr(api, "graph", GraphWithConversationStore())
    monkeypatch.setattr(api.profile_summarizer, "refresh", calls.append)

    with TestClient(api.app) as client:
        response = client.post("/chat", json={"user_id": "U1001", "message": "我平时喜欢徒步"})

    assert response.status_code == 200
    assert calls == ["U1001"]


def test_gradio_refreshes_profile_only_after_saved_response(tmp_path):
    from blood_agents.web import ChatApplication

    app = ChatApplication(str(tmp_path / "profile-web.db"))
    calls = []

    class ProfileRefreshSpy:
        def refresh(self, user_id):
            calls.append(user_id)
            return {"status": "up_to_date", "through_turn_id": 0}

    app.profile_summarizer = ProfileRefreshSpy()
    try:
        app.refresh_session_title("U1001", "s", {
            "user_id": "U1001", "session_id": "s", "trace_id": "trace-saved",
            "conversation_persistence": {"status": "saved"},
        })
        assert calls == ["U1001"]
        app.refresh_session_title("U1002", "other", {
            "user_id": "U1001", "session_id": "s",
            "conversation_persistence": {"status": "saved"},
        })
        assert calls == ["U1001"]
    finally:
        app.graph.close()


def test_profile_schema_migrates_existing_tables_without_dropping_rows(tmp_path):
    path = tmp_path / "profile-old-schema.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE user_profile_summaries(user_id TEXT PRIMARY KEY,profile_json TEXT NOT NULL,"
            "summarized_through_turn_id INTEGER NOT NULL,updated_at TEXT);"
            "CREATE TABLE user_profile_events(id INTEGER PRIMARY KEY,user_id TEXT,profile_key TEXT,"
            "action TEXT,value TEXT,source_session_id TEXT,source_turn_id INTEGER,change_index INTEGER,"
            "created_at TEXT);"
            "INSERT INTO user_profile_summaries VALUES('U1001','{\"hobbies\":[\"徒步\"]}',1,'now');"
        )

    repo = BloodRepository(str(path))

    assert repo.get_user_profile("U1001") == {"hobbies": ["徒步"]}
    for table in ("user_profile_summaries", "user_profile_events"):
        with repo.connection() as conn:
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert "extractor_version" in columns
