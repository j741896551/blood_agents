"""确保 SQLAlchemy saver 可以无损读取旧 SQLite 文件并遵守 LangGraph 契约。"""

import sqlite3

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from blood_agents.storage.checkpoint import SQLAlchemySqliteSaver


def test_existing_sqlite_saver_file_is_readable_without_replay(tmp_path):
    """先用原 SqliteSaver 写入，再用新实现恢复快照、元数据和待处理写入。"""
    path = tmp_path / "old.checkpoints.db"
    connection = sqlite3.connect(path)
    original = SqliteSaver(connection)
    config = {"configurable": {"thread_id": "user-session", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["appointment_state"] = {"status": "awaiting_confirmation"}
    saved = original.put(config, checkpoint, {"source": "input", "step": 1}, {})
    original.put_writes(saved, [("appointment", {"site": "大学城献血车"})], "task-one")
    connection.close()

    saver = SQLAlchemySqliteSaver(path)
    try:
        restored = saver.get_tuple(config)
        assert restored.checkpoint["channel_values"] == checkpoint["channel_values"]
        assert restored.metadata["source"] == "input"
        assert restored.pending_writes == [
            ("task-one", "appointment", {"site": "大学城献血车"}),
        ]
        assert [item.config["configurable"]["checkpoint_id"] for item in saver.list(
            config, filter={"step": 1}, limit=1,
        )] == [checkpoint["id"]]
        saver.delete_thread("user-session")
        assert saver.get_tuple(config) is None
        assert list(saver.list(config)) == []
    finally:
        saver.close()


def test_sqlalchemy_saver_writes_are_readable_by_original_saver(tmp_path):
    """新格式继续兼容原 saver，便于在迁移前做隔离回滚验证。"""
    path = tmp_path / "new.checkpoints.db"
    saver = SQLAlchemySqliteSaver(path)
    config = {"configurable": {"thread_id": "session", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    try:
        saved = saver.put(config, checkpoint, {"source": "loop", "step": 2}, {})
        saver.put_writes(saved, [("result", "ok")], "task-one")
    finally:
        saver.close()

    with sqlite3.connect(path) as connection:
        original = SqliteSaver(connection)
        restored = original.get_tuple(config)
        assert restored.checkpoint["id"] == checkpoint["id"]
        assert restored.metadata["step"] == 2
        assert restored.pending_writes == [("task-one", "result", "ok")]


def test_metadata_filter_rejects_unsafe_key(tmp_path):
    """元数据过滤键必须受限，避免构造任意 JSON 路径或 SQL。"""
    saver = SQLAlchemySqliteSaver(tmp_path / "filtered.db")
    try:
        with pytest.raises(ValueError, match="invalid metadata filter"):
            list(saver.list(None, filter={"step'; DROP TABLE checkpoints; --": 1}))
    finally:
        saver.close()
