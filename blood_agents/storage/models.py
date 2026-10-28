"""Agent 自有数据的 SQLAlchemy 映射；外部业务表和 LangGraph checkpoint 不在这里。"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """集中持有 Agent 数据表元数据，供建表和后续迁移工具读取。"""


class ConversationTurn(Base):
    """一轮已完成的用户与助手对话；历史正文在仓储边界脱敏。"""

    __tablename__ = "conversation_turns"
    __table_args__ = (Index("idx_conversation_turns_session", "user_id", "session_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    user_id: Mapped[str] = mapped_column(String(80), nullable=False)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_message: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(80), nullable=False, server_default=text("''"))
    entrypoint: Mapped[str] = mapped_column(String(80), nullable=False, server_default=text("''"))
    response_mode: Mapped[str] = mapped_column(String(80), nullable=False, server_default=text("''"))
    created_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                             server_default=text("CURRENT_TIMESTAMP"))


class ConversationSessionSummary(Base):
    """某个用户、某个会话的标题及摘要。"""

    __tablename__ = "conversation_session_summaries"

    user_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    summarized_through_turn_id: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger: Mapped[str] = mapped_column(String(80), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                             server_default=text("CURRENT_TIMESTAMP"))


class ConversationDeletionPending(Base):
    """跨会话库和 checkpoint 删除时的持久化墓碑。"""

    __tablename__ = "conversation_deletion_pending"

    user_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    requested_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                               server_default=text("CURRENT_TIMESTAMP"))


class PreferenceEvent(Base):
    """用户明确设置或撤销偏好的事件，不保存模型推断的业务事实。"""

    __tablename__ = "preference_events"
    __table_args__ = (
        CheckConstraint("action IN ('set', 'revoke')"),
        Index("idx_preference_events_user_key", "user_id", "preference_key", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(80), nullable=False)
    preference_key: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[str | None] = mapped_column(Text)
    source_session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_trace_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    source_quote: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                             server_default=text("CURRENT_TIMESTAMP"))


class UserProfileSummary(Base):
    """按 user_id 保存的偏好画像快照和提取游标。"""

    __tablename__ = "user_profile_summaries"

    user_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    profile_json: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'{}'"))
    summarized_through_turn_id: Mapped[int] = mapped_column(Integer, nullable=False,
                                                              server_default=text("0"))
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False,
                                                    server_default=text("''"))
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                             server_default=text("CURRENT_TIMESTAMP"))


class UserProfileEvent(Base):
    """画像变更的来源轮次；只保存白名单字段。"""

    __tablename__ = "user_profile_events"
    __table_args__ = (
        CheckConstraint("profile_key IN ('hobby')"),
        CheckConstraint("action IN ('set', 'revoke', 'clear')"),
        UniqueConstraint("user_id", "source_turn_id", "change_index"),
        Index("idx_user_profile_events_user_source", "user_id", "source_turn_id", "change_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(80), nullable=False)
    profile_key: Mapped[str] = mapped_column(String(16), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[str | None] = mapped_column(Text)
    source_session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_turn_id: Mapped[int] = mapped_column(Integer, nullable=False)
    change_index: Mapped[int] = mapped_column(Integer, nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(80), nullable=False,
                                                    server_default=text("''"))
    created_at: Mapped[str] = mapped_column(String(32), nullable=False,
                                             server_default=text("CURRENT_TIMESTAMP"))
