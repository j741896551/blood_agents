"""入口身份与会话联合隔离；这是本地状态隔离，不是登录认证。"""

import hashlib
import json
import uuid
from threading import Lock
from time import monotonic
from weakref import finalize

from blood_agents.domains.points import CONFIRM_WORDS
from blood_agents.log.trace_logging import TraceContext, current_trace, trace_event


class StalePendingActionError(ValueError):
    """按钮对应的待确认预览已变化；拒绝执行任何图节点或业务工具。"""


class ConversationGraph:
    def __init__(self, graph, conversation_store=None, checkpoint_connection=None):
        """初始化 ConversationGraph 的依赖与运行状态。"""
        self.graph = graph
        self.conversation_store = conversation_store
        # 文件 checkpointer 的 SQLAlchemy 引擎与编译图同寿命，可显式释放连接池。
        self._checkpoint_finalizer = (
            finalize(self, checkpoint_connection.close) if checkpoint_connection is not None else None
        )
        self._guard = Lock()
        self._locks = {}

    def close(self) -> None:
        """释放本实例持有的 checkpoint 引擎；不删除状态文件。"""
        if self._checkpoint_finalizer is not None and self._checkpoint_finalizer.alive:
            self._checkpoint_finalizer()

    @staticmethod
    def _thread_key(user_id: str, session_id: str) -> str:
        """将用户和会话联合编码，生成隔离的 checkpoint 键。"""
        # 与 invoke 的 thread_id 映射共用一处，避免删除误选其他用户的 checkpoint。
        return hashlib.sha256(json.dumps([user_id, session_id]).encode()).hexdigest()

    def _session_lock(self, key: str) -> Lock:
        """取得当前用户会话专用的线程锁。"""
        with self._guard:
            return self._locks.setdefault(key, Lock())

    @staticmethod
    def _appointment_button_from_snapshot(snapshot, user_id: str, session_id: str) -> dict | None:
        """只读取已交付的预约预览；checkpoint ID 是按钮的版本号。"""
        if snapshot is None:
            return None
        values = snapshot.values or {}
        slots = values.get("appointment_state") or {}
        status = (values.get("result") or {}).get("status")
        if values.get("active_agent") != "appointment" or not slots.get("preview_delivered"):
            return None
        if status == "awaiting_confirmation" and slots.get("status") == "awaiting_confirmation":
            operation, confirm_text, abandon_text = "book", "确认预约", "退出预约"
            confirm_label = "确认预约"
        elif status == "awaiting_action_confirmation":
            operation = slots.get("pending_operation")
            if operation == "cancel_appointment":
                confirm_text, abandon_text, confirm_label = "确认取消", "放弃操作", "确认取消预约"
            elif operation == "checkin_appointment":
                confirm_text, abandon_text, confirm_label = "确认签到", "放弃操作", "确认签到"
            else:
                return None
        else:
            return None
        checkpoint_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
        if not checkpoint_id:
            return None
        # 不把内部目标预约号、身份或原始 checkpoint 交给网页；任何新轮次都会换版本。
        token = hashlib.sha256(json.dumps(
            [user_id, session_id, checkpoint_id, operation], ensure_ascii=False,
        ).encode()).hexdigest()
        return {"token": token, "operation": operation, "confirm_text": confirm_text,
                "abandon_text": abandon_text, "confirm_label": confirm_label}

    @staticmethod
    def _points_button_from_snapshot(snapshot, user_id: str, session_id: str) -> dict | None:
        """只放行已由模型交付、且当前轮仍为同一积分操作的完整预览。"""
        if snapshot is None:
            return None
        values = snapshot.values or {}
        pending = values.get("points_state") or {}
        result = values.get("result") or {}
        kind = pending.get("kind")
        status = ("awaiting_redemption_confirmation" if kind == "redeem"
                  else "awaiting_points_confirmation")
        modes = values.get("model_info") or {}
        if (values.get("active_agent") != "points" or kind not in CONFIRM_WORDS
                or result.get("status") != status or result.get("operation") != kind
                or result.get("confirm_word") != CONFIRM_WORDS[kind]
                or (modes.get("navigation") or {}).get("mode") != "llm"
                or (modes.get("points") or {}).get("mode") != "llm"
                or (values.get("response_info") or {}).get("mode") != "llm"):
            return None
        checkpoint_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
        if not checkpoint_id:
            return None
        operation = f"points:{kind}"
        token = hashlib.sha256(json.dumps(
            [user_id, session_id, checkpoint_id, operation], ensure_ascii=False,
        ).encode()).hexdigest()
        return {"token": token, "operation": operation, "confirm_text": CONFIRM_WORDS[kind],
                "abandon_text": "放弃操作", "confirm_label": CONFIRM_WORDS[kind]}

    @classmethod
    def _pending_button_from_snapshot(cls, snapshot, user_id: str, session_id: str) -> dict | None:
        """从 checkpoint 快照提取当前可点击的待确认操作。"""
        return (cls._appointment_button_from_snapshot(snapshot, user_id, session_id)
                or cls._points_button_from_snapshot(snapshot, user_id, session_id))

    def get_pending_button(
        self, user_id: str, session_id: str, *, expected_state: dict | None = None,
    ) -> dict | None:
        """页面读预约/积分按钮状态；用户/会话联合隔离，不调用工具。"""
        user_id, session_id = user_id.strip(), session_id.strip()
        if not user_id or not session_id:
            return None
        key = self._thread_key(user_id, session_id)
        with self._session_lock(key):
            pending_check = getattr(self.conversation_store, "is_conversation_session_deletion_pending", None)
            if callable(pending_check) and pending_check(user_id, session_id):
                return None
            snapshot = self.graph.get_state({"configurable": {"thread_id": key}})
            if snapshot is None:
                return None
            if expected_state is not None and (
                (snapshot.values or {}).get("appointment_state") != expected_state.get("appointment_state")
                or (snapshot.values or {}).get("result") != expected_state.get("result")
            ):
                # 另一个并发轮次已更新 checkpoint；旧回复不能挂上新预览的按钮。
                return None
            return self._pending_button_from_snapshot(snapshot, user_id, session_id)

    def get_pending_appointment_button(
        self, user_id: str, session_id: str, *, expected_state: dict | None = None,
    ) -> dict | None:
        """保留预约专用读取入口，避免积分操作被旧调用方误认成预约。"""
        pending = self.get_pending_button(user_id, session_id, expected_state=expected_state)
        return pending if pending and not pending["operation"].startswith("points:") else None

    def _finish_pending_deletion(self, user_id: str, session_id: str, key: str) -> int:
        """调用方已持有会话锁；失败时保留墓碑，下次先重试，不运行旧状态。"""
        checkpointer = getattr(self.graph, "checkpointer", None)
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if not callable(delete_thread):
            raise TypeError("checkpoint deletion is unavailable")
        delete_thread(key)
        return self.conversation_store.finish_conversation_session_deletion(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> int:
        """仅供可信本地调用；不暴露给没有身份认证的网页或 API。"""
        user_id, session_id = user_id.strip(), session_id.strip()
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        if self.conversation_store is None:
            raise RuntimeError("conversation store is unavailable")
        if not callable(getattr(getattr(self.graph, "checkpointer", None), "delete_thread", None)):
            raise TypeError("checkpoint deletion is unavailable")
        key = self._thread_key(user_id, session_id)
        with self._session_lock(key):
            self.conversation_store.begin_conversation_session_deletion(user_id, session_id)
            # 两库没有原子事务：墓碑必须先提交，再删 checkpoint，最后清对话表。
            # 任一后续步骤失败都不能移除墓碑，防止下一轮恢复待确认写操作。
            return self._finish_pending_deletion(user_id, session_id, key)

    def invoke(self, state, config=None, *, expected_pending: dict | None = None):
        """执行一轮会话，并把入口会话键转换为用户隔离的 checkpoint 键。

        session_id 是前端可复用的会话标识，thread_id 是图内部状态索引；
        trace_id 每轮重新生成，只用于关联本轮日志，不用于恢复会话。

        Args:
            state: 包含本轮消息和入口 user_id 的图输入。
            config: 包含原始 session_id 的 LangGraph 调用配置。
            expected_pending: 点击确认/放弃按钮时要求匹配的待确认版本。

        Returns:
            图执行状态，并附加本轮 trace_id 和对话落库状态。
        """
        config = dict(config or {})
        configurable = dict(config.get("configurable", {}))
        user_id = state.get("user_id", "").strip()
        session_id = str(configurable.get("thread_id", "")).strip()
        if not user_id or not session_id:
            raise ValueError("user_id and session_id required")
        # JSON 编码避免拼接分隔符碰撞；不同用户即使使用相同会话 ID 也隔离。
        key = self._thread_key(user_id, session_id)
        configurable["thread_id"] = key
        config["configurable"] = configurable
        entrypoint = config.get("metadata", {}).get("entrypoint", "graph")
        trace_id = uuid.uuid4().hex
        context = TraceContext(trace_id, user_id, session_id, entrypoint)
        token = current_trace.set(context)
        message = state.get("messages", [])
        question = message[-1].content if message else None
        started = monotonic()
        lock = self._session_lock(key)
        try:
            trace_event("turn.start", summary={"question_length": len(question) if isinstance(question, str) else 0},
                        full={"user_question": question})
            # 同一会话串行更新 checkpoint，避免同时确认/修改造成状态覆盖。
            with lock:
                trace_event("session.lock_acquired")
                pending_check = getattr(
                    self.conversation_store, "is_conversation_session_deletion_pending", None,
                )
                if callable(pending_check) and pending_check(user_id, session_id):
                    # 清理失败后绝不能用旧 checkpoint 继续；成功重试后才执行新消息。
                    self._finish_pending_deletion(user_id, session_id, key)
                    trace_event("conversation.deletion_recovered")
                if expected_pending is not None:
                    # 点击确认/放弃仍是一次普通图输入，但必须在同一会话锁内核对
                    # 身份、待确认操作和 checkpoint 版本，防止旧按钮、双击与改参后误写。
                    current = self._pending_button_from_snapshot(
                        self.graph.get_state(config), user_id, session_id,
                    )
                    action = expected_pending.get("action")
                    expected_text = current.get(f"{action}_text") if current else None
                    if (not current or action not in {"confirm", "abandon"}
                            or expected_pending.get("token") != current["token"]
                            or expected_pending.get("operation") != current["operation"]
                            or question != expected_text):
                        trace_event("ui.pending_action_stale", summary={"action": action})
                        raise StalePendingActionError("待确认操作已变化，请重新查看最新预览")
                # 原始会话 ID 只由入口提供；长期记忆需要它记录来源，不让模型补造。
                result = self.graph.invoke(
                    {**state, "user_id": user_id, "session_id": session_id}, config=config,
                )
                answer = result.get("messages", [])
                answer_text = answer[-1].content if answer else None
                persistence = {"status": "disabled"}
                if self.conversation_store is not None:
                    try:
                        if not isinstance(question, str) or not isinstance(answer_text, str):
                            raise TypeError("conversation turn requires text messages")
                # 对话正文用于历史展示与摘要，checkpoint 用于恢复图状态；
                # 二者不能互相重放。落库与图执行共用锁，保证轮次顺序。
                        self.conversation_store.save_conversation_turn(
                            trace_id=trace_id, user_id=user_id, session_id=session_id,
                            user_message=question, assistant_message=answer_text,
                            intent=str(result.get("intent") or ""), entrypoint=str(entrypoint),
                            response_mode=str(result.get("response_info", {}).get("mode") or ""),
                        )
                        persistence = {"status": "saved"}
                        trace_event("conversation.saved")
                    except Exception as exc:  # noqa: BLE001 - 写入失败不能重放已执行的业务工具
                        persistence = {"status": "failed", "error_type": type(exc).__name__}
                        trace_event("conversation.save_failed", summary={"error_type": type(exc).__name__})
            business_result = result.get("result", {})
            trace_event("turn.end", summary={
                "intent": result.get("intent"),
                "result_status": business_result.get("status"),
                "result_code": business_result.get("code"),
                "model_mode": result.get("model_info", {}).get("mode"),
                "reply_mode": result.get("response_info", {}).get("mode"),
                "node_path": [item["node"] for item in result.get("node_trace", [])],
                "node_count": len(result.get("node_trace", [])),
                "tool_count": len(result.get("tool_trace", [])),
                "conversation_persistence": persistence["status"],
                "duration_ms": round((monotonic() - started) * 1000, 2),
            }, full={"answer": answer_text})
            # 返回本轮 ID 供网页/API 调试区和 JSONL 日志互相定位；不写入 checkpoint。
            return {**result, "trace_id": trace_id, "conversation_persistence": persistence}
        except Exception as exc:
            trace_event("turn.error", summary={
                "error_type": type(exc).__name__,
                "duration_ms": round((monotonic() - started) * 1000, 2),
            })
            raise
        finally:
            current_trace.reset(token)
