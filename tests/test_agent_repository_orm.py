"""只验证 Agent 自有六张表；不依赖外部业务替身或平台模型。"""

import sqlite3

import pytest
from sqlalchemy import inspect

from blood_agents.storage.models import Base
from blood_agents.storage.repository import AgentRepository


def _turn(repo, trace, user="U1001", conversation="s"):
    """建立一轮不含敏感资料的对话。"""
    repo.save_conversation_turn(
        trace_id=trace, user_id=user, session_id=conversation,
        user_message="喜欢徒步", assistant_message="已收到", intent="general", entrypoint="test",
    )
    return repo.list_conversation_turns(user, conversation)[-1]["id"]


def test_orm_models_cover_only_agent_tables_and_reopen_existing_file(tmp_path):
    """模型映射、重启读取和既有文件不丢数据。"""
    path = tmp_path / "agent.db"
    repo = AgentRepository(str(path))
    assert set(Base.metadata.tables) == {
        "conversation_turns", "conversation_session_summaries", "conversation_deletion_pending",
        "preference_events", "user_profile_summaries", "user_profile_events",
    }
    assert set(Base.metadata.tables) <= set(inspect(repo.engine).get_table_names())
    turn_id = _turn(repo, "first")
    assert repo.save_conversation_summary(
        user_id="U1001", session_id="s", title="爱好咨询", summary="用户谈及徒步爱好。",
        through_turn_id=turn_id, trigger="first_reply",
    )
    assert repo.save_preference_event(
        user_id="U1001", key="preferred_site", action="set", value="大学城献血车",
        source_session_id="s", source_trace_id="preference-1", source_quote="更喜欢大学城献血车",
    )
    assert repo.save_user_profile_update(
        user_id="U1001", through_turn_id=turn_id, profile={"hobbies": ["徒步"]},
        changes=[{"key": "hobby", "action": "set", "value": "徒步",
                  "source_session_id": "s", "source_turn_id": turn_id}],
        extractor_version="test-v1",
    )

    reopened = AgentRepository(str(path))
    assert reopened.list_conversation_sessions("U1001")[0]["title"] == "爱好咨询"
    assert reopened.get_user_profile("U1001") == {"hobbies": ["徒步"]}
    assert reopened.list_user_profile_events("U1001")[0]["source_turn_id"] == turn_id
    assert reopened.list_active_preferences("U1001")[0]["value"] == "大学城献血车"
    assert not reopened.save_preference_event(
        user_id="U1001", key="preferred_site", action="set", value="大学城献血车",
        source_session_id="s", source_trace_id="preference-1", source_quote="更喜欢大学城献血车",
    )
    reopened.begin_conversation_session_deletion("U1001", "s")
    reopened.begin_conversation_session_deletion("U1001", "s")
    assert reopened.list_conversation_turns("U1001", "s") == []
    with pytest.raises(RuntimeError, match="deletion is pending"):
        _turn(reopened, "blocked")
    assert reopened.finish_conversation_session_deletion("U1001", "s") == 1
    assert reopened.list_conversation_sessions("U1001") == []
    # 会话删除不应擅自删除独立维护的长期偏好与画像。
    assert reopened.get_user_profile("U1001") == {"hobbies": ["徒步"]}


def test_existing_agent_schema_is_upgraded_without_replacing_rows(tmp_path):
    """旧 SQLite 少字段时只补字段，不重建表，也不动额外旧业务表。"""
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE conversation_turns(id INTEGER PRIMARY KEY,trace_id TEXT UNIQUE,"
            "user_id TEXT,session_id TEXT,user_message TEXT,assistant_message TEXT,"
            "intent TEXT DEFAULT '',entrypoint TEXT DEFAULT '',created_at TEXT DEFAULT CURRENT_TIMESTAMP);"
            "INSERT INTO conversation_turns(trace_id,user_id,session_id,user_message,"
            "assistant_message) VALUES('old','U1001','s','你好','您好');"
            "CREATE TABLE user_profile_summaries(user_id TEXT PRIMARY KEY,profile_json TEXT,"
            "summarized_through_turn_id INTEGER,updated_at TEXT);"
            "INSERT INTO user_profile_summaries VALUES('U1001','{\"hobbies\":[\"徒步\"]}',1,'now');"
            "CREATE TABLE user_profile_events(id INTEGER PRIMARY KEY,user_id TEXT,"
            "profile_key TEXT,action TEXT,value TEXT,source_session_id TEXT,"
            "source_turn_id INTEGER,change_index INTEGER,created_at TEXT);"
            "CREATE TABLE appointments(id INTEGER PRIMARY KEY,marker TEXT);"
            "INSERT INTO appointments(marker) VALUES('keep');"
        )
    repo = AgentRepository(str(path))
    assert repo.list_conversation_turns("U1001", "s")[0]["response_mode"] == ""
    assert repo.get_user_profile("U1001") == {"hobbies": ["徒步"]}
    for table, column in (
        ("conversation_turns", "response_mode"),
        ("user_profile_summaries", "extractor_version"),
        ("user_profile_events", "extractor_version"),
    ):
        assert column in {item["name"] for item in inspect(repo.engine).get_columns(table)}
    assert "idx_conversation_turns_session" in {
        item["name"] for item in inspect(repo.engine).get_indexes("conversation_turns")
    }
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT marker FROM appointments").fetchone()[0] == "keep"
