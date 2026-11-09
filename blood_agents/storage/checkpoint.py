"""用 SQLAlchemy 管理 LangGraph 的同步 SQLite checkpoint。

表名、列名和序列化格式与项目此前使用的 SqliteSaver 3.0.3 一致，
因此原有独立 checkpoint 文件可以直接打开，不需要重放对话或搬迁数据。
这仍是本地 SQLite 实现，不提供跨进程会话锁或生产级并发保证。
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy import LargeBinary, String, cast, create_engine, delete, event, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class CheckpointBase(DeclarativeBase):
    """独立 checkpoint 文件的表元数据，不混入 Agent 会话仓储。"""


class StoredCheckpoint(CheckpointBase):
    """一个 LangGraph 状态快照；联合主键沿用官方 SQLite saver。"""

    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(String, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String, primary_key=True, default="")
    checkpoint_id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String)
    type: Mapped[str | None] = mapped_column(String)
    checkpoint: Mapped[bytes | None] = mapped_column(LargeBinary)
    metadata_blob: Mapped[bytes | None] = mapped_column("metadata", LargeBinary)


class StoredWrite(CheckpointBase):
    """节点执行过程中的待处理写入，必须与快照一起恢复。"""

    __tablename__ = "writes"

    thread_id: Mapped[str] = mapped_column(String, primary_key=True)
    checkpoint_ns: Mapped[str] = mapped_column(String, primary_key=True, default="")
    checkpoint_id: Mapped[str] = mapped_column(String, primary_key=True)
    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    idx: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str | None] = mapped_column(String)
    value: Mapped[bytes | None] = mapped_column(LargeBinary)


class SQLAlchemySqliteSaver(BaseCheckpointSaver[str]):
    """LangGraph 同步 saver：通过 ORM 会话读写原 SQLite checkpoint 表。"""

    def __init__(self, path: str | Path) -> None:
        """打开独立 SQLite 文件并建立缺失的表，不改写已有 checkpoint。"""
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            URL.create("sqlite+pysqlite", database=str(self.path)),
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )

        @event.listens_for(self.engine, "connect")
        def _set_busy_timeout(connection, _record):
            """每条连接设置等待时间，避免短暂写锁直接报错。"""
            connection.execute("PRAGMA busy_timeout=30000")

        try:
            with self.engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            CheckpointBase.metadata.create_all(self.engine)
            self.path.chmod(0o600)
        except Exception:
            self.engine.dispose()
            raise
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def close(self) -> None:
        """释放引擎连接池；不删除持久化状态。"""
        self.engine.dispose()

    def _tuple(self, session, row: StoredCheckpoint) -> CheckpointTuple:
        """把原始快照和关联写入还原为 LangGraph 约定的数据结构。"""
        config = {"configurable": {
            "thread_id": row.thread_id, "checkpoint_ns": row.checkpoint_ns,
            "checkpoint_id": row.checkpoint_id,
        }}
        writes = session.scalars(select(StoredWrite).where(
            StoredWrite.thread_id == row.thread_id,
            StoredWrite.checkpoint_ns == row.checkpoint_ns,
            StoredWrite.checkpoint_id == row.checkpoint_id,
        ).order_by(StoredWrite.task_id, StoredWrite.idx)).all()
        parent = ({"configurable": {
            "thread_id": row.thread_id, "checkpoint_ns": row.checkpoint_ns,
            "checkpoint_id": row.parent_checkpoint_id,
        }} if row.parent_checkpoint_id else None)
        return CheckpointTuple(
            config,
            self.serde.loads_typed((row.type, row.checkpoint)),
            json.loads(row.metadata_blob) if row.metadata_blob is not None else {},
            parent,
            [(item.task_id, item.channel, self.serde.loads_typed((item.type, item.value)))
             for item in writes],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """按 thread/命名空间和可选版本取快照；无版本时取最新。"""
        values = config["configurable"]
        statement = select(StoredCheckpoint).where(
            StoredCheckpoint.thread_id == str(values["thread_id"]),
            StoredCheckpoint.checkpoint_ns == values.get("checkpoint_ns", ""),
        )
        if checkpoint_id := get_checkpoint_id(config):
            statement = statement.where(StoredCheckpoint.checkpoint_id == checkpoint_id)
        else:
            statement = statement.order_by(StoredCheckpoint.checkpoint_id.desc()).limit(1)
        with self._sessions() as session:
            row = session.scalar(statement)
            return self._tuple(session, row) if row is not None else None

    def list(
        self, config: RunnableConfig | None,
        *, filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """按官方 saver 的筛选顺序列出历史版本和待处理写入。"""
        statement = select(StoredCheckpoint)
        if config is not None:
            values = config["configurable"]
            statement = statement.where(StoredCheckpoint.thread_id == str(values["thread_id"]))
            if values.get("checkpoint_ns") is not None:
                statement = statement.where(StoredCheckpoint.checkpoint_ns == values["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                statement = statement.where(StoredCheckpoint.checkpoint_id == checkpoint_id)
        if filter:
            for key, value in filter.items():
                if not re.fullmatch(r"[a-zA-Z0-9_.-]+", key):
                    raise ValueError(f"invalid metadata filter key: {key!r}")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, separators=(",", ":"))
                elif isinstance(value, bool):
                    value = int(value)
                elif value is not None and not isinstance(value, (str, int, float)):
                    value = str(value)
                statement = statement.where(
                    func.json_extract(cast(StoredCheckpoint.metadata_blob, String), f"$.{key}") == value
                )
        if before is not None:
            statement = statement.where(StoredCheckpoint.checkpoint_id < get_checkpoint_id(before))
        statement = statement.order_by(StoredCheckpoint.checkpoint_id.desc())
        if limit is not None:
            statement = statement.limit(limit)
        with self._sessions() as session:
            rows = session.scalars(statement).all()
            result = [self._tuple(session, row) for row in rows]
        return iter(result)

    def put(
        self, config: RunnableConfig, checkpoint: Checkpoint,
        metadata: CheckpointMetadata, new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """原子保存新快照；重复 checkpoint ID 只更新对应快照。"""
        values = config["configurable"]
        kind, payload = self.serde.dumps_typed(checkpoint)
        row = {
            "thread_id": str(values["thread_id"]), "checkpoint_ns": values["checkpoint_ns"],
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": values.get("checkpoint_id"),
            "type": kind, "checkpoint": payload,
            "metadata": json.dumps(
                get_checkpoint_metadata(config, metadata), ensure_ascii=False,
            ).encode("utf-8", "ignore"),
        }
        statement = sqlite_insert(StoredCheckpoint).values(**row)
        statement = statement.on_conflict_do_update(
            index_elements=["thread_id", "checkpoint_ns", "checkpoint_id"],
            set_={key: value for key, value in row.items() if key not in {
                "thread_id", "checkpoint_ns", "checkpoint_id",
            }},
        )
        with self._sessions.begin() as session:
            session.execute(statement)
        return {"configurable": {
            "thread_id": values["thread_id"], "checkpoint_ns": values["checkpoint_ns"],
            "checkpoint_id": checkpoint["id"],
        }}

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]],
        task_id: str, task_path: str = "",
    ) -> None:
        """保存节点待处理写入；特殊通道可覆盖，其他通道重复写入忽略。"""
        values = config["configurable"]
        replace = all(channel in WRITES_IDX_MAP for channel, _ in writes)
        with self._sessions.begin() as session:
            for index, (channel, value) in enumerate(writes):
                kind, payload = self.serde.dumps_typed(value)
                row = {
                    "thread_id": str(values["thread_id"]),
                    "checkpoint_ns": str(values["checkpoint_ns"]),
                    "checkpoint_id": str(values["checkpoint_id"]),
                    "task_id": task_id, "idx": WRITES_IDX_MAP.get(channel, index),
                    "channel": channel, "type": kind, "value": payload,
                }
                statement = sqlite_insert(StoredWrite).values(**row)
                if replace:
                    statement = statement.on_conflict_do_update(
                        index_elements=["thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"],
                        set_={key: row[key] for key in ("channel", "type", "value")},
                    )
                else:
                    statement = statement.on_conflict_do_nothing()
                session.execute(statement)

    def delete_thread(self, thread_id: str) -> None:
        """在一个事务中删除指定 thread 的快照和写入。"""
        with self._sessions.begin() as session:
            session.execute(delete(StoredWrite).where(StoredWrite.thread_id == str(thread_id)))
            session.execute(delete(StoredCheckpoint).where(StoredCheckpoint.thread_id == str(thread_id)))

    def get_next_version(self, current: str | None, channel: None) -> str:
        """保持与既有 SqliteSaver 相同的单调版本格式。"""
        current_number = 0 if current is None else (
            current if isinstance(current, int) else int(current.split(".")[0])
        )
        return f"{current_number + 1:032}.{random.random():016}"
