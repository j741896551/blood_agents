"""Agent 自有数据仓储；用 SQLAlchemy ORM 访问六张会话与记忆表。"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import and_, create_engine, delete, event, exists, func, inspect, select
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased, sessionmaker
from sqlalchemy.pool import StaticPool

from blood_agents.storage.models import (
    Base,
    ConversationDeletionPending,
    ConversationSessionSummary,
    ConversationTurn,
    PreferenceEvent,
    UserProfileEvent,
    UserProfileSummary,
)
from blood_agents.storage.privacy import sanitize_conversation_text


class AgentRepository:
    """保存 Agent 对话、摘要、偏好和画像；外部业务数据与 checkpoint 由其他组件管理。"""

    def __init__(self, db_path: str = ":memory:", *, conversation_retention_days: int | None = None) -> None:
        """创建 SQLite 引擎和会话工厂，复用原有文件和表名。"""
        self.db_path = db_path
        if conversation_retention_days is None:
            conversation_retention_days = int(os.getenv("BLOOD_AGENT_CONVERSATION_RETENTION_DAYS", "0"))
        if conversation_retention_days < 0:
            raise ValueError("conversation_retention_days must be non-negative")
        self.conversation_retention_days = conversation_retention_days
        # SQLite 本地演示对短事务串行化；跨进程/多 Worker 并发要在数据库迁移阶段另验。
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        options = {"connect_args": {"check_same_thread": False, "timeout": 30.0}}
        if db_path == ":memory:":
            # 同一个内存数据库供 ORM 和仍需底层连接的兼容调用共同使用。
            options["poolclass"] = StaticPool
        self.engine = create_engine(URL.create("sqlite+pysqlite", database=db_path), **options)

        @event.listens_for(self.engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record):
            """每个 SQLite 连接都启用外键约束。"""
            dbapi_connection.execute("PRAGMA foreign_keys = ON")

        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self.init_schema()
        if db_path != ":memory:":
            # 已存在的对话文件同样收紧权限，不清理或重建原数据。
            Path(db_path).chmod(0o600)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """提供原子 ORM 事务；数据库操作结束后才释放本地互斥锁。"""
        with self._lock, self._session_factory.begin() as session:
            yield session

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """兼容原有只在测试/旧调用中使用的 SQLite 底层连接。

        AgentRepository 自身的读写均通过 session() 和映射模型完成。该入口暂留给
        已有测试的旧数据注入及手工检查，未来迁移 MySQL 前应替换这些直接 SQL 调用。
        """
        with self._lock:
            connection = self.engine.raw_connection()
            connection.driver_connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def init_schema(self) -> None:
        """依据模型建缺失表，并无损补齐旧版 SQLite 的三个新增字段。"""
        with self._lock:
            Base.metadata.create_all(self.engine)
            # 旧版本在这些字段出现前已建表；create_all 不会 ALTER 已有表。
            # 此处只做最小兼容 DDL，后续正式版本迁移交给 Alembic。
            with self.engine.begin() as connection:
                for table, column in (
                    ("conversation_turns", "response_mode"),
                    ("user_profile_summaries", "extractor_version"),
                    ("user_profile_events", "extractor_version"),
                ):
                    columns = {item["name"] for item in inspect(connection).get_columns(table)}
                    if column not in columns:
                        connection.exec_driver_sql(
                            f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                        )
                # create_all 不会为已存在的表补建后来新增的索引。
                for table in Base.metadata.tables.values():
                    for index in table.indexes:
                        index.create(bind=connection, checkfirst=True)

    @staticmethod
    def _pending(session: Session, user_id: str, session_id: str) -> bool:
        """检查跨 checkpoint 删除的持久墓碑。"""
        return session.get(ConversationDeletionPending, (user_id, session_id)) is not None

    def save_conversation_turn(
        self, *, trace_id: str, user_id: str, session_id: str,
        user_message: str, assistant_message: str, intent: str, entrypoint: str,
        response_mode: str = "",
    ) -> None:
        """保存一轮已完成对话；写入前脱敏，并按配置清理过期会话元数据。"""
        safe_user_message = sanitize_conversation_text(user_message)
        safe_assistant_message = sanitize_conversation_text(assistant_message)
        with self.session() as session:
            if self._pending(session, user_id, session_id):
                raise RuntimeError("conversation session deletion is pending")
            if self.conversation_retention_days:
                cutoff = (datetime.now(timezone.utc) - timedelta(
                    days=self.conversation_retention_days,
                )).strftime("%Y-%m-%d %H:%M:%S")
                session.execute(delete(ConversationTurn).where(ConversationTurn.created_at < cutoff))
                # 对话都已过期的会话不能留下标题/摘要；偏好画像并不随此策略清理。
                session.execute(delete(ConversationSessionSummary).where(~exists(
                    select(ConversationTurn.id).where(
                        ConversationTurn.user_id == ConversationSessionSummary.user_id,
                        ConversationTurn.session_id == ConversationSessionSummary.session_id,
                    )
                )))
            session.add(ConversationTurn(
                trace_id=trace_id, user_id=user_id, session_id=session_id,
                user_message=safe_user_message, assistant_message=safe_assistant_message,
                intent=intent, entrypoint=entrypoint, response_mode=response_mode,
            ))

    def list_conversation_turns(
        self, user_id: str, session_id: str, *, limit: int = 50,
    ) -> list[dict]:
        """按登录用户和会话取最近轮次，再按发生顺序返回。"""
        if not user_id or not session_id or not 1 <= limit <= 100:
            raise ValueError("user_id、session_id 非空且 limit 为 1 到 100")
        with self.session() as session:
            if self._pending(session, user_id, session_id):
                return []
            turns = session.scalars(
                select(ConversationTurn).where(
                    ConversationTurn.user_id == user_id,
                    ConversationTurn.session_id == session_id,
                ).order_by(ConversationTurn.id.desc()).limit(limit)
            ).all()
            return [
                {
                    "id": turn.id, "trace_id": turn.trace_id, "user_id": turn.user_id,
                    "session_id": turn.session_id,
                    "user_message": sanitize_conversation_text(turn.user_message),
                    "assistant_message": sanitize_conversation_text(turn.assistant_message),
                    "intent": turn.intent, "entrypoint": turn.entrypoint,
                    "response_mode": turn.response_mode, "created_at": turn.created_at,
                }
                for turn in reversed(turns)
            ]

    def list_user_conversation_turns_after(
        self, user_id: str, after_turn_id: int, *, limit: int = 100,
    ) -> list[dict]:
        """按用户增量读取画像提取所需的用户文字，排除待删除会话。"""
        if not user_id or after_turn_id < 0 or not 1 <= limit <= 500:
            raise ValueError("valid user, cursor and limit required")
        pending = exists(select(ConversationDeletionPending.user_id).where(
            ConversationDeletionPending.user_id == ConversationTurn.user_id,
            ConversationDeletionPending.session_id == ConversationTurn.session_id,
        ))
        with self.session() as session:
            turns = session.scalars(select(ConversationTurn).where(
                ConversationTurn.user_id == user_id,
                ConversationTurn.id > after_turn_id,
                ~pending,
            ).order_by(ConversationTurn.id).limit(limit)).all()
            return [
                {
                    "id": turn.id, "trace_id": turn.trace_id, "session_id": turn.session_id,
                    "user_message": sanitize_conversation_text(turn.user_message),
                    "response_mode": turn.response_mode, "created_at": turn.created_at,
                }
                for turn in turns
            ]

    def get_user_profile(self, user_id: str) -> dict:
        """读取画像白名单内容；姓名、血型等业务资料不进入这里。"""
        if not user_id:
            raise ValueError("user_id required")
        with self.session() as session:
            row = session.get(UserProfileSummary, user_id)
            profile_json = row.profile_json if row else None
        if not profile_json:
            return {"hobbies": []}
        try:
            profile = json.loads(profile_json)
        except (TypeError, json.JSONDecodeError):
            return {"hobbies": []}
        if not isinstance(profile, dict):
            return {"hobbies": []}
        hobbies = profile.get("hobbies", [])
        if not isinstance(hobbies, list):
            return {"hobbies": []}
        safe_hobbies = [
            item for item in hobbies[:20]
            if isinstance(item, str) and 1 <= len(item) <= 40
            and sanitize_conversation_text(item) == item
        ]
        return {"hobbies": safe_hobbies}

    def get_user_profile_cursor(self, user_id: str) -> int:
        """读取该用户画像已覆盖的对话轮次 ID。"""
        if not user_id:
            raise ValueError("user_id required")
        with self.session() as session:
            row = session.get(UserProfileSummary, user_id)
            return int(row.summarized_through_turn_id) if row else 0

    def save_user_profile_update(
        self, *, user_id: str, through_turn_id: int, profile: dict, changes: list[dict],
        extractor_version: str = "",
    ) -> bool:
        """单事务保存画像事件和快照；迟到提取不得回滚游标。"""
        if not user_id or through_turn_id < 1 or not isinstance(profile, dict):
            raise ValueError("valid user profile update required")
        if set(profile) - {"hobbies"}:
            raise ValueError("unsupported user profile field")
        if not isinstance(changes, list):
            raise TypeError("profile changes must be a list")
        hobbies = profile.get("hobbies", [])
        if (not isinstance(hobbies, list) or len(hobbies) > 20
                or any(not isinstance(item, str) or not 1 <= len(item) <= 40
                       or sanitize_conversation_text(item) != item for item in hobbies)):
            raise ValueError("invalid hobby profile values")
        profile_json = json.dumps({"hobbies": list(dict.fromkeys(hobbies))}, ensure_ascii=False)
        with self.session() as session:
            old = session.get(UserProfileSummary, user_id)
            previous_cursor = int(old.summarized_through_turn_id) if old else 0
            if through_turn_id <= previous_cursor:
                return False
            for index, change in enumerate(changes):
                key, action, value = change.get("key"), change.get("action"), change.get("value")
                source_session_id = change.get("source_session_id")
                source_turn_id = change.get("source_turn_id")
                if (key != "hobby" or action not in {"set", "revoke", "clear"}
                        or not source_session_id or not isinstance(source_turn_id, int)
                        or not previous_cursor < source_turn_id <= through_turn_id
                        or (action == "clear" and value is not None)
                        or (action in {"set", "revoke"} and not isinstance(value, str))):
                    raise ValueError("invalid user profile event")
                if self._pending(session, user_id, source_session_id):
                    raise ValueError("profile event source turn does not belong to user")
                source = session.scalar(select(ConversationTurn.id).where(
                    ConversationTurn.id == source_turn_id,
                    ConversationTurn.user_id == user_id,
                    ConversationTurn.session_id == source_session_id,
                ))
                if source is None:
                    raise ValueError("profile event source turn does not belong to user")
                if (value is not None and (
                        not isinstance(value, str) or not 1 <= len(value) <= 40
                        or sanitize_conversation_text(value) != value)):
                    raise ValueError("profile event contains private text")
                session.add(UserProfileEvent(
                    user_id=user_id, profile_key=key, action=action, value=value,
                    source_session_id=source_session_id, source_turn_id=source_turn_id,
                    change_index=index, extractor_version=extractor_version,
                ))
            if old is None:
                session.add(UserProfileSummary(
                    user_id=user_id, profile_json=profile_json,
                    summarized_through_turn_id=through_turn_id,
                    extractor_version=extractor_version,
                ))
            else:
                old.profile_json = profile_json
                old.summarized_through_turn_id = through_turn_id
                old.extractor_version = extractor_version
                old.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return True

    def list_user_profile_events(self, user_id: str) -> list[dict]:
        """返回画像变更的来源元数据，不暴露原始对话片段。"""
        if not user_id:
            raise ValueError("user_id required")
        with self.session() as session:
            events = session.scalars(select(UserProfileEvent).where(
                UserProfileEvent.user_id == user_id,
            ).order_by(UserProfileEvent.source_turn_id, UserProfileEvent.change_index,
                       UserProfileEvent.id)).all()
            return [
                {
                    "profile_key": item.profile_key, "action": item.action, "value": item.value,
                    "source_session_id": item.source_session_id,
                    "source_turn_id": item.source_turn_id, "change_index": item.change_index,
                    "extractor_version": item.extractor_version, "created_at": item.created_at,
                }
                for item in events
            ]

    def list_conversation_sessions(self, user_id: str, *, limit: int = 30) -> list[dict]:
        """列出指定用户最近活动的会话，以及对应标题摘要。"""
        if not user_id or not 1 <= limit <= 100:
            raise ValueError("user_id 非空且 limit 为 1 到 100")
        pending = exists(select(ConversationDeletionPending.user_id).where(
            ConversationDeletionPending.user_id == ConversationTurn.user_id,
            ConversationDeletionPending.session_id == ConversationTurn.session_id,
        ))
        grouped = select(
            ConversationTurn.session_id.label("session_id"),
            func.min(ConversationTurn.id).label("first_id"),
            func.max(ConversationTurn.id).label("last_id"),
            func.count().label("turn_count"),
        ).where(ConversationTurn.user_id == user_id, ~pending).group_by(
            ConversationTurn.session_id,
        ).order_by(func.max(ConversationTurn.id).desc()).limit(limit).subquery()
        first_turn = aliased(ConversationTurn)
        last_turn = aliased(ConversationTurn)
        stmt = select(
            grouped.c.session_id, grouped.c.turn_count, grouped.c.last_id,
            first_turn.user_message.label("first_message"),
            last_turn.created_at.label("updated_at"),
            ConversationSessionSummary.title, ConversationSessionSummary.summary,
            ConversationSessionSummary.summarized_through_turn_id.label("summary_turn_id"),
            ConversationSessionSummary.trigger.label("summary_trigger"),
        ).join(first_turn, first_turn.id == grouped.c.first_id).join(
            last_turn, last_turn.id == grouped.c.last_id,
        ).outerjoin(ConversationSessionSummary, and_(
            ConversationSessionSummary.user_id == user_id,
            ConversationSessionSummary.session_id == grouped.c.session_id,
        )).order_by(grouped.c.last_id.desc())
        with self.session() as session:
            rows = session.execute(stmt).mappings().all()
            return [
                {
                    **dict(row),
                    "first_message": sanitize_conversation_text(row["first_message"]),
                    "title": sanitize_conversation_text(row["title"]) if row["title"] else None,
                    "summary": sanitize_conversation_text(row["summary"]) if row["summary"] else None,
                }
                for row in rows
            ]

    def save_conversation_summary(
        self, *, user_id: str, session_id: str, title: str, summary: str,
        through_turn_id: int, trigger: str,
    ) -> bool:
        """只在游标对应当前会话最新轮次时，保存较新的摘要。"""
        if not user_id or not session_id or not isinstance(through_turn_id, int) or through_turn_id < 1:
            raise ValueError("valid user, session and turn id required")
        if (sanitize_conversation_text(title) != title
                or sanitize_conversation_text(summary) != summary):
            raise ValueError("conversation summary contains private data")
        with self.session() as session:
            if self._pending(session, user_id, session_id):
                return False
            latest = session.scalar(select(func.max(ConversationTurn.id)).where(
                ConversationTurn.user_id == user_id,
                ConversationTurn.session_id == session_id,
            ))
            if latest != through_turn_id:
                return False
            old = session.get(ConversationSessionSummary, (user_id, session_id))
            if old and through_turn_id <= old.summarized_through_turn_id:
                return False
            if old is None:
                session.add(ConversationSessionSummary(
                    user_id=user_id, session_id=session_id, title=title, summary=summary,
                    summarized_through_turn_id=through_turn_id, trigger=trigger,
                ))
            else:
                old.title, old.summary, old.trigger = title, summary, trigger
                old.summarized_through_turn_id = through_turn_id
                old.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return True

    def delete_conversation_session(self, user_id: str, session_id: str) -> int:
        """只删除 Agent 仓储中的指定会话；checkpoint 由调用方单独处理。"""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        with self.session() as session:
            count = session.execute(delete(ConversationTurn).where(
                ConversationTurn.user_id == user_id,
                ConversationTurn.session_id == session_id,
            )).rowcount
            session.execute(delete(ConversationSessionSummary).where(
                ConversationSessionSummary.user_id == user_id,
                ConversationSessionSummary.session_id == session_id,
            ))
            return count

    def begin_conversation_session_deletion(self, user_id: str, session_id: str) -> None:
        """先持久化删除墓碑，避免 checkpoint 删除失败时重新恢复旧状态。"""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        with self.session() as session:
            if not self._pending(session, user_id, session_id):
                session.add(ConversationDeletionPending(user_id=user_id, session_id=session_id))

    def is_conversation_session_deletion_pending(self, user_id: str, session_id: str) -> bool:
        """检查指定会话的删除墓碑是否存在。"""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        with self.session() as session:
            return self._pending(session, user_id, session_id)

    def finish_conversation_session_deletion(self, user_id: str, session_id: str) -> int:
        """checkpoint 清理成功后，在同一事务清对话、摘要与墓碑。"""
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        with self.session() as session:
            if not self._pending(session, user_id, session_id):
                raise RuntimeError("conversation deletion was not started")
            count = session.execute(delete(ConversationTurn).where(
                ConversationTurn.user_id == user_id,
                ConversationTurn.session_id == session_id,
            )).rowcount
            session.execute(delete(ConversationSessionSummary).where(
                ConversationSessionSummary.user_id == user_id,
                ConversationSessionSummary.session_id == session_id,
            ))
            session.execute(delete(ConversationDeletionPending).where(
                ConversationDeletionPending.user_id == user_id,
                ConversationDeletionPending.session_id == session_id,
            ))
            return count

    def delete_user_conversation_history(self, user_id: str) -> int:
        """删除一个用户的对话与摘要；入口身份校验由调用方负责。"""
        if not user_id:
            raise ValueError("user_id required")
        with self.session() as session:
            count = session.execute(delete(ConversationTurn).where(
                ConversationTurn.user_id == user_id,
            )).rowcount
            session.execute(delete(ConversationSessionSummary).where(
                ConversationSessionSummary.user_id == user_id,
            ))
            return count

    def save_preference_event(
        self, *, user_id: str, key: str, action: str, value: str | None,
        source_session_id: str, source_trace_id: str, source_quote: str,
    ) -> bool:
        """只追加用户明确表达的偏好事件，相同 trace_id 幂等。"""
        if not user_id or not source_session_id or not source_trace_id:
            raise ValueError("user, session and trace are required")
        if key not in {"preferred_site", "preferred_time_slot"} or action not in {"set", "revoke"}:
            raise ValueError("unsupported preference event")
        if action == "set" and not value or action == "revoke" and value is not None:
            raise ValueError("invalid preference value")
        if not source_quote or sanitize_conversation_text(source_quote) != source_quote:
            raise ValueError("preference source contains private data")
        if value is not None and sanitize_conversation_text(value) != value:
            raise ValueError("preference value contains private data")
        with self.session() as session:
            if session.scalar(select(PreferenceEvent.id).where(
                PreferenceEvent.source_trace_id == source_trace_id,
            )) is not None:
                return False
            # SAVEPOINT 保证其他进程先写入相同 trace_id 时本事务仍可正常结束。
            try:
                with session.begin_nested():
                    session.add(PreferenceEvent(
                        user_id=user_id, preference_key=key, action=action, value=value,
                        source_session_id=source_session_id, source_trace_id=source_trace_id,
                        source_quote=source_quote,
                    ))
                    session.flush()
            except IntegrityError:
                return False
            return True

    def list_active_preferences(self, user_id: str) -> list[dict]:
        """只返回每个偏好键的最后一条且仍有效的事件。"""
        if not user_id:
            raise ValueError("user_id required")
        latest = select(func.max(PreferenceEvent.id)).where(
            PreferenceEvent.user_id == user_id,
        ).group_by(PreferenceEvent.preference_key)
        with self.session() as session:
            events = session.scalars(select(PreferenceEvent).where(
                PreferenceEvent.user_id == user_id,
                PreferenceEvent.action == "set",
                PreferenceEvent.id.in_(latest),
            ).order_by(PreferenceEvent.preference_key)).all()
            return [
                {
                    "preference_key": item.preference_key, "value": item.value,
                    "source_session_id": item.source_session_id,
                    "source_trace_id": item.source_trace_id,
                    "source_quote": item.source_quote, "created_at": item.created_at,
                }
                for item in events
            ]
