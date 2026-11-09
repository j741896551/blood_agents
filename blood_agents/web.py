from __future__ import annotations

import os
import uuid
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Optional

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
existing_no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
local_no_proxy = ",".join(part for part in (existing_no_proxy, "127.0.0.1", "localhost") if part)
os.environ["NO_PROXY"] = local_no_proxy
os.environ["no_proxy"] = local_no_proxy

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .llm.conversation_summarizer import ConversationSummarizer
from .llm.reply_composer import reply_stream_callback
from .log.trace_logging import TraceContext, current_trace, mask_sensitive, trace_event
from .orchestration.conversation import StalePendingActionError
from .orchestration.graph import build_conversational_graph
from .orchestration.router import IntentRouter
from .orchestration.user_profile import UserProfileSummarizer, is_profile_query
from .storage.privacy import REDACTED_MESSAGE
from .storage.repository import AgentRepository
from .ui.presentation import EMPTY_SUMMARY, execution_summary

NEW_CONVERSATION_WELCOME = """你好！很高兴见到你。我可以帮你了解以下演示能力：

- **献血预约**：预约献血、查询预约记录、取消已提交预约、预约签到，以及补充或退出当前预约
- **积分与礼品**：查询积分、推荐和兑换礼品，管理兑换订单与收货地址
- **献血记录**：查询个人献血记录与历史；下次献血日期可结合献血间隔规则和个人记录一起参考
- **献血知识**：回答献血相关知识

需要说明的是，目前使用的是本地 Mock/SQLite 演示数据，未接入真实公司的旧业务接口；OCR 仅保留原型，当前暂缓，暂不支持直接上传图片识别。想先了解哪一项呢？"""


def _new_conversation_history() -> list[tuple[None, str]]:
    """只在 Gradio 展示欢迎引导；不作为用户轮次提交、持久化或生成摘要。"""
    return [(None, NEW_CONVERSATION_WELCOME)]


def _mask_debug_donor(value):
    """网页调试区不能回显用户为代约输入的完整手机号。"""
    if isinstance(value, dict):
        return {key: (item[:3] + "****" + item[-4:]
                      if key == "donor_mobile" and isinstance(item, str) and len(item) == 11
                      else _mask_debug_donor(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_debug_donor(item) for item in value]
    return value


class ChatApplication:
    """Gradio UI 与 LangGraph 之间的薄适配层。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """初始化 ChatApplication 的依赖与运行状态。"""
        load_dotenv()
        self.repo = AgentRepository(db_path or os.getenv("BLOOD_AGENT_DB", "./blood_agent.db"))
        router = IntentRouter.from_env()
        self.graph = build_conversational_graph(self.repo, router=router)
        # 标题生成复用业务模型网关的主备/并发控制；失败只影响侧栏元数据。
        self.summarizer = ConversationSummarizer(router.response_llm)
        self.profile_summarizer = UserProfileSummarizer(self.repo, router.response_llm)

    def respond(
        self,
        message: str,
        history: list[tuple[str | None, str]],
        user_id: str,
        session_id: str,
        *,
        expected_pending: dict | None = None,
    ) -> tuple[str, list[tuple[str | None, str]], str, dict]:
        """处理一次网页消息；session_id 会映射为 LangGraph thread_id。

        history 只用于当前 Gradio 组件显示，跨轮业务状态由图 checkpoint
        恢复；不能靠页面上可见的聊天记录重建待确认预约或兑换。

        Args:
            message: 用户本轮输入，按钮确认也会转换为一次输入。
            history: 仅用于页面渲染的已显示对话。
            user_id: 演示入口指定的用户编号。
            session_id: 前端会话编号，同一会话的下一轮应复用。
            expected_pending: 按钮携带的待确认快照，用于拒绝过期点击。

        Returns:
            清空输入框的值、更新后的页面历史、会话编号和脱敏调试信息。
        """
        message = message.strip()
        user_id = user_id.strip()
        session_id = session_id.strip() or uuid.uuid4().hex
        history = history or []
        if not user_id:
            answer = "请先填写用户 ID。演示账号可以使用 U1001。"
            return "", history + [(message, answer)], session_id, {"error": "missing_user_id"}
        if not message:
            return "", history, session_id, {}
        try:
            # 若上一轮后台画像更新失败，用户询问画像时先增量重试，再进入图读取。
            if is_profile_query(message):
                self._refresh_user_profile(user_id, session_id)
            # 网页只传本轮用户消息和入口身份；图从 checkpoint 恢复以前的槽位。
            # 同一 session_id 因此能继续预约/兑换，不需要页面重新提交全部历史。
            payload = {"messages": [HumanMessage(content=message)], "user_id": user_id}
            config = {"configurable": {"thread_id": session_id},
                      "metadata": {"entrypoint": "gradio"}}
            state = (self.graph.invoke(payload, config=config, expected_pending=expected_pending)
                     if expected_pending is not None else self.graph.invoke(payload, config=config))
            answer = state["messages"][-1].content
            # 只有当前轮实际产生可确认预览才显示按钮；读取 checkpoint 失败
            # 不能把已经执行的写操作重试，也不能抹掉本轮真实回执。
            try:
                reader = getattr(self.graph, "get_pending_button", None)
                pending = reader(user_id, session_id, expected_state=state) if callable(reader) else None
                if state.get("result", {}).get("status") not in {
                    "awaiting_confirmation", "awaiting_action_confirmation",
                    "awaiting_redemption_confirmation", "awaiting_points_confirmation",
                }:
                    pending = None
            except Exception:  # noqa: BLE001 - 按钮只影响交互，不影响已完成业务轮次
                pending = None
            details = {
                "trace_id": state.get("trace_id"),
                "session_id": session_id,
                "user_id": user_id,
                "intent": state.get("intent"),
                "result": mask_sensitive(state.get("result", {})),
                "slots": mask_sensitive(_mask_debug_donor(state.get("slots", {}))),
                "understanding": mask_sensitive(_mask_debug_donor(state.get("understanding", {}))),
                "active_agent": state.get("active_agent"),
                "agent_status": state.get("agent_status"),
                "agent_handoff": state.get("agent_handoff", {}),
                "model_info": state.get("model_info", {}),
                "response_info": state.get("response_info", {}),
                "tool_plan_info": state.get("tool_plan_info", {}),
                "tool_trace": state.get("tool_trace", []),
                "node_trace": state.get("node_trace", []),
                "conversation_persistence": state.get("conversation_persistence", {}),
                "pending_action": pending,
            }
        except StalePendingActionError:
            answer = "操作按钮已失效，请查看最新待确认信息后再操作。"
            details = {"ui_action": {"status": "stale"}}
            return "", history + [(None, answer)], session_id, details
        except Exception as exc:  # noqa: BLE001 - UI boundary must return a safe error
            answer = "服务暂时不可用，请稍后重试。"
            details = {"error": type(exc).__name__}
        return "", history + [(mask_sensitive(message), mask_sensitive(answer))], session_id, details

    def respond_with_summary(self, message, history, user_id, session_id):
        """完成网页对话并更新标题摘要展示。"""
        result = self.respond(message, history, user_id, session_id)
        return (*result, execution_summary(result[3]))

    def respond_with_stream(self, message, history, user_id, session_id):
        """Gradio 生成器：只要主导航模型主动发出普通回复片段，就逐步更新气泡。"""
        events: Queue = Queue(maxsize=128)
        stopped = Event()

        def publish_chunk(chunk: str) -> None:
            """把普通导航回复的流式片段放入页面事件队列。"""
            # 用户断开页面后不再堆积事件；底层供应商请求是否能及时取消取决于 SDK。
            while not stopped.is_set():
                try:
                    events.put(("chunk", chunk), timeout=0.1)
                    return
                except Full:
                    continue

        def publish_final(payload) -> None:
            """向前端发布最终校验后的回复。"""
            while not stopped.is_set():
                try:
                    events.put(("final", payload), timeout=0.1)
                    return
                except Full:
                    continue

        def run_turn() -> None:
            """执行固定话术中的一轮对话并记录结果。"""
            token = reply_stream_callback.set(publish_chunk)
            try:
                publish_final(self.respond_with_summary(
                    message, history, user_id, session_id,
                ))
            except Exception:  # noqa: BLE001 - UI 边界返回安全错误，不输出异常原文
                publish_final(("", (history or []) + [
                    (message, "服务暂时不可用，请稍后重试。")
                ], session_id, {"error": "stream_worker_failed"}, EMPTY_SUMMARY))
            finally:
                reply_stream_callback.reset(token)

        Thread(target=run_turn, daemon=True, name="gradio-chat-stream").start()
        visible_history = list(history or [])
        partial = ""
        try:
            while True:
                try:
                    kind, value = events.get(timeout=0.25)
                except Empty:
                    # 图可能在意图识别/工具阶段运行；不伪造“正在生成”的文本。
                    continue
                if kind == "chunk":
                    partial += value
                    current = visible_history + [(mask_sensitive(message), partial)]
                    yield "", current, session_id, {}, "正在流式生成普通导航回复……"
                    continue
                yield value
                break
        finally:
            stopped.set()

    def _pending_button(self, action: str, history, user_id, session_id, details):
        """按钮走原有对话图；不相信页面文字或旧按钮携带的业务结果。"""
        user_id, session_id = (user_id or "").strip(), (session_id or "").strip()
        details = details if isinstance(details, dict) else {}
        pending = details.get("pending_action") or {}
        pending = pending if isinstance(pending, dict) else {}
        if (not user_id or not session_id or details.get("user_id") != user_id
                or details.get("session_id") != session_id or not pending.get("token")):
            result = ("", (history or []) + [(None, "操作按钮已失效，请查看最新待确认信息后再操作。")],
                      session_id, {"ui_action": {"status": "stale"}})
        else:
            text = pending.get(f"{action}_text", "")
            result = self.respond(text, history, user_id, session_id, expected_pending={
                "token": pending["token"], "operation": pending.get("operation"), "action": action,
            })
        return (*result, execution_summary(result[3]))

    def confirm_pending_button(self, history, user_id, session_id, details):
        """把页面上的确认按钮转成一轮受校验的会话输入。"""
        return self._pending_button("confirm", history, user_id, session_id, details)

    def abandon_pending_button(self, history, user_id, session_id, details):
        """把页面上的放弃按钮转成一轮受校验的会话输入。"""
        return self._pending_button("abandon", history, user_id, session_id, details)

    @staticmethod
    def pending_button_updates(details):
        """根据当前待确认状态更新页面按钮。"""
        pending = details.get("pending_action") if isinstance(details, dict) else None
        pending = pending if isinstance(pending, dict) else {}
        visible = bool(pending.get("token"))
        confirm_label = pending.get("confirm_label", "确认")
        # 这是页面操作说明而非业务答复；与服务端确认词同源，模型漏提按钮时仍能找到入口。
        hint = (f"请核对上方预览。点击「{confirm_label}」提交，或点击「放弃本次操作」；"
                f"也可以在输入框回复「{pending.get('confirm_text', confirm_label)}」。")
        return (gr.update(visible=visible, value=hint),
                gr.update(visible=visible, value=confirm_label),
                gr.update(visible=visible))

    @staticmethod
    def _session_choice(row: dict) -> tuple[str, str]:
        """标题来自首轮问答的模型摘要；未生成时绝不退回首句。"""
        title = row.get("title") or "待生成标题"
        if "消息内容未保存" in title:
            title = "待生成标题"
        return f"{title} · {row['turn_count']} 轮 · {row['session_id'][-6:]}", row["session_id"]

    @classmethod
    def _session_radio(cls, rows: list[dict], selected_id: str | None = None):
        """根据已保存会话生成侧栏选择项。"""
        choices = [cls._session_choice(row) for row in rows]
        if selected_id and selected_id not in {value for _, value in choices}:
            # 刚创建但尚未发送消息的会话只存在于页面，不伪称已经写入数据库。
            choices.insert(0, ("新对话（尚未保存）", selected_id))
        return gr.update(choices=choices, value=selected_id)

    def load_user_sessions(self, user_id: str):
        """页面首次打开/切换用户时，默认展示该用户最近的已保存会话。"""
        user_id = (user_id or "").strip()
        if not user_id:
            return self._session_radio([]), "", [], {}, "请先填写用户 ID。", ""
        try:
            rows = self.repo.list_conversation_sessions(user_id)
            session_id = rows[0]["session_id"] if rows else uuid.uuid4().hex
            history, details, summary = (
                self.restore_history(user_id, session_id) if rows
                else (_new_conversation_history(), {},
                      "新对话：发送第一条消息后会出现在左侧记录中。")
            )
            return self._session_radio(rows, session_id), session_id, history, details, summary, ""
        except Exception as exc:  # noqa: BLE001 - UI 不能泄露原始数据库报错
            session_id = uuid.uuid4().hex
            return self._session_radio([], session_id), session_id, [], {
                "session_list": {"status": "failed", "error_type": type(exc).__name__},
            }, "读取对话列表失败，请稍后刷新。", ""

    def new_session(self, user_id: str):
        """仅创建前端会话 ID；不会删除任何旧对话或 checkpoint。"""
        session_id = uuid.uuid4().hex
        try:
            rows = self.repo.list_conversation_sessions((user_id or "").strip()) if user_id else []
        except Exception:  # noqa: BLE001 - 新对话不应因侧栏读取失败而不可用
            rows = []
        return (self._session_radio(rows, session_id), session_id,
                _new_conversation_history(), {},
                "新对话：旧记录仍保留。", "")

    def select_session(self, user_id: str, session_id: str):
        """侧栏选中项须属于当前填写的用户；只读加载，不调用图。"""
        user_id = (user_id or "").strip()
        session_id = (session_id or "").strip()
        if not user_id or not session_id:
            return (self._session_radio([]), "", [],
                    {"session_list": {"status": "missing_identity"}}, "请先选择会话。", "")
        try:
            rows = self.repo.list_conversation_sessions(user_id)
            if session_id not in {row["session_id"] for row in rows}:
                return (self._session_radio(rows), "", [],
                        {"session_list": {"status": "not_found"}}, "该用户没有这条对话。", "")
            history, details, summary = self.restore_history(user_id, session_id)
            return self._session_radio(rows, session_id), session_id, history, details, summary, ""
        except Exception as exc:  # noqa: BLE001 - 不把 DB 内部错误显示在页面
            return self._session_radio([]), "", [], {"session_list": {
                "status": "failed", "error_type": type(exc).__name__,
            }}, "读取对话失败，请稍后重试。", ""

    def restore_with_sidebar(self, user_id: str, session_id: str):
        """保留手动输入会话 ID 的演示入口，同时同步侧栏选中状态。"""
        history, details, summary = self.restore_history(user_id, session_id)
        try:
            rows = self.repo.list_conversation_sessions((user_id or "").strip()) if user_id else []
        except Exception:  # noqa: BLE001 - 历史读取结果仍可展示
            rows = []
        return self._session_radio(rows, (session_id or "").strip() or None), history, details, summary

    def manual_session_input(self, user_id: str, session_id: str):
        """手动改会话 ID 时先清除旧对话，待用户显式加载历史。"""
        try:
            rows = self.repo.list_conversation_sessions((user_id or "").strip()) if user_id else []
        except Exception:  # noqa: BLE001 - 清屏必须照常执行
            rows = []
        return self._session_radio(rows, (session_id or "").strip() or None), [], {}, EMPTY_SUMMARY

    def respond_with_sidebar(self, message, history, user_id, session_id):
        """完成网页对话并同步侧栏会话列表。"""
        result = self.respond_with_summary(message, history, user_id, session_id)
        return (*result, self.refresh_session_title(user_id, result[2], result[3]))

    def _ensure_session_digest(self, user_id: str, session_id: str,
                               *, trace_id: str | None = None) -> None:
        """首次打开或首次回复时按需补摘要；已有标题不再额外调用模型。"""
        rows = self.repo.list_conversation_sessions(user_id)
        row = next((item for item in rows if item["session_id"] == session_id), None)
        if not row or (row.get("title") and row.get("summary_trigger") != "static_cache"):
            return
        turns = self.repo.list_conversation_turns(user_id, session_id, limit=100)
        if not turns:
            return
        # 纯缓存会话若再为标题送模，就违背“唯一命中不调用模型”；用审核过的
        # 通用展示标题。以后出现非缓存轮次，再走原本的模型摘要路径。
        if all(turn.get("response_mode") == "static_cache" for turn in turns):
            self.repo.save_conversation_summary(
                user_id=user_id, session_id=session_id,
                title="演示系统标准问答", summary="本会话仅包含演示系统范围的固定标准问答。",
                through_turn_id=turns[-1]["id"], trigger="static_cache",
            )
            return
        context = TraceContext(trace_id or uuid.uuid4().hex, user_id, session_id, "gradio_title")
        token = current_trace.set(context)
        try:
            digest, _ = self.summarizer.summarize(turns)
            if digest:
                saved = self.repo.save_conversation_summary(
                    user_id=user_id, session_id=session_id,
                    title=digest.title, summary=digest.summary,
                    through_turn_id=turns[-1]["id"],
                    trigger="first_reply" if len(turns) == 1 and trace_id else "legacy_open",
                )
                trace_event("conversation.digest.saved", summary={
                    "saved": saved, "through_turn_id": turns[-1]["id"],
                })
        finally:
            current_trace.reset(token)

    def refresh_session_title(self, user_id: str, session_id: str, details: dict):
        """聊天回复已交给页面后生成标题；标题故障不改变本轮答复。"""
        user_id, session_id = (user_id or "").strip(), (session_id or "").strip()
        details = details or {}
        try:
            if (user_id and session_id
                    and details.get("user_id") == user_id
                    and details.get("session_id") == session_id
                    and details.get("conversation_persistence", {}).get("status") == "saved"):
                self._refresh_user_profile(user_id, session_id, details.get("trace_id"))
                self._ensure_session_digest(user_id, session_id,
                                            trace_id=details.get("trace_id"))
            rows = self.repo.list_conversation_sessions(user_id) if user_id else []
        except Exception:  # noqa: BLE001 - 标题生成失败不能抹掉已有会话列表
            try:
                rows = self.repo.list_conversation_sessions(user_id) if user_id else []
            except Exception:  # noqa: BLE001 - 页面边界不泄露数据库异常
                rows = []
        return self._session_radio(rows, session_id)

    def _refresh_user_profile(self, user_id: str, session_id: str,
                               trace_id: str | None = None) -> None:
        """在答复交付后的 UI 事件中更新画像，模型失败不影响已经完成的对话。"""
        context = TraceContext(trace_id or uuid.uuid4().hex, user_id, session_id, "gradio_profile")
        token = current_trace.set(context)
        try:
            outcome = self.profile_summarizer.refresh(user_id)
            trace_event("user_profile.refresh.after_turn", summary={
                "status": outcome.get("status"),
                "through_turn_id": outcome.get("through_turn_id"),
                "candidate_turn_count": outcome.get("candidate_turn_count"),
            })
        except Exception as exc:  # noqa: BLE001 - 画像只属个性化体验，不能影响回复
            trace_event("user_profile.refresh.failed", summary={"error_type": type(exc).__name__})
        finally:
            current_trace.reset(token)

    def summarize_loaded_session(self, user_id: str, session_id: str):
        """旧会话只在被打开时补标题，不在页面启动时批量请求模型。"""
        user_id, session_id = (user_id or "").strip(), (session_id or "").strip()
        try:
            if user_id and session_id:
                self._refresh_user_profile(user_id, session_id)
                self._ensure_session_digest(user_id, session_id)
            rows = self.repo.list_conversation_sessions(user_id) if user_id else []
        except Exception:  # noqa: BLE001 - 历史回显已经完成，摘要失败仅保留占位标题
            try:
                rows = self.repo.list_conversation_sessions(user_id) if user_id else []
            except Exception:  # noqa: BLE001 - 页面边界不泄露数据库异常
                rows = []
        return self._session_radio(rows, session_id)

    def restore_history(self, user_id: str, session_id: str):
        """只读回显已保存轮次；流程状态由下次新消息的 checkpoint 加载。"""
        user_id = (user_id or "").strip()
        session_id = (session_id or "").strip()
        if not user_id or not session_id:
            return [], {"history_restore": {"status": "missing_identity"}}, "请先填写用户 ID 和会话 ID。"
        try:
            # 仓储强制按用户和会话联合查询，并对历史文本再次做隐私过滤。
            turns = self.repo.list_conversation_turns(user_id, session_id, limit=100)
        except Exception as exc:  # noqa: BLE001 - 页面边界不暴露数据库报错或旧数据
            return [], {"history_restore": {
                "status": "failed", "error_type": type(exc).__name__,
            }}, "读取历史对话失败，请稍后重试。"
        history = [(turn["user_message"], turn["assistant_message"]) for turn in turns]
        status = "loaded" if history else "empty"
        details = {"user_id": user_id, "session_id": session_id, "history_restore": {
            "status": status, "count": len(history), "limit": 100,
            "user_id": user_id, "session_id": session_id,
        }}
        reader = getattr(self.graph, "get_pending_button", None)
        if (callable(reader) and turns and turns[-1].get("response_mode") == "llm"
                and turns[-1].get("assistant_message") != REDACTED_MESSAGE):
            # 恢复后只见脱敏占位符，不等于用户已看到可核对的预览；不展示确认按钮。
            try:
                details["pending_action"] = reader(user_id, session_id)
            except Exception:  # noqa: BLE001 - 旧消息回显不依赖按钮状态读取
                details["pending_action"] = None
        summary = f"已加载最近 {len(history)} 轮脱敏对话；本次未调用模型或业务接口。"
        return history, details, summary


def build_ui(chat_app: Optional[ChatApplication] = None) -> gr.Blocks:
    """构建页面和事件绑定，实际业务仍由 LangGraph 处理。"""
    chat_app = chat_app or ChatApplication()
    with gr.Blocks(
        title="血站智慧献血导航", theme=gr.themes.Soft(),
        css=("#conversation-list { max-height: 65vh; overflow-y: auto; }"
             "#conversation-actions { padding: 8px 14px 12px; border-top: 1px solid #d9dfe7; }"
             "#conversation-actions .prose { margin-bottom: 4px; }"),
    ) as demo:
        gr.Markdown(
            """
            # 🩸 血站智慧献血导航
            通过自然语言办理献血预约、预约查询/取消/签到、积分查询、礼品兑换、献血记录查询和知识问答。
            当前使用本地 Mock 旧业务接口；`U1001` 可从空预约开始，`U1002` 有四状态合成旧记录。
            """
        )
        with gr.Row():
            with gr.Column(scale=2, min_width=260):
                user_id = gr.Textbox(label="用户 ID", value="U1001",
                                     info="本地演示身份，可填写其他用户 ID")
                clear = gr.Button("新对话", variant="primary")
                refresh = gr.Button("刷新对话列表")
                sessions = gr.Radio(label="最近对话", choices=[], elem_id="conversation-list",
                                    info="按最近回复排序；只显示当前用户的已保存轮次")
                with gr.Accordion("手动输入会话 ID（调试）", open=False):
                    session_id = gr.Textbox(label="会话 ID", value=lambda: uuid.uuid4().hex,
                                            info="切换左侧对话时会自动更新")
                    restore = gr.Button("加载此会话历史")
            with gr.Column(scale=7, min_width=420):
                # 对话与操作按钮放在同一卡片；按钮紧贴最新回复，位于输入框上方。
                with gr.Group(elem_id="conversation-card"):
                    # Gradio 5 仍支持成对消息；显式固定类型以保持 respond() 的历史契约。
                    chatbot = gr.Chatbot(label="对话", height=480, type="tuples")
                    with gr.Column(elem_id="conversation-actions"):
                        pending_hint = gr.Markdown(visible=False)
                        with gr.Row():
                            confirm_pending = gr.Button("确认", variant="primary", visible=False)
                            abandon_pending = gr.Button("放弃本次操作", visible=False)
                with gr.Row():
                    message = gr.Textbox(
                        label="请输入消息", placeholder="例如：我要预约", lines=2, scale=8,
                    )
                    send = gr.Button("发送", variant="primary", scale=1)
                summary = gr.Textbox(label="本轮执行摘要", value=EMPTY_SUMMARY,
                                     lines=12, interactive=False)
                with gr.Accordion("原始编排数据（演示/调试）", open=False):
                    details = gr.JSON(label="本轮编排信息（演示/调试）", open=False)
                gr.Examples(
                    examples=[
                        ["我要预约"], ["还是上次网点，选择明天上午"], ["确认预约"],
                        ["查询我的预约信息"], ["帮我取消明天下午市中心献血屋的预约"],
                        ["帮我签到后天上午大学城献血车的预约"], ["查询我的积分"],
                        ["查询我的献血记录"], ["献血前可以空腹吗？"],
                    ],
                    inputs=[message],
                )

        submit_inputs = [message, chatbot, user_id, session_id]
        submit_outputs = [message, chatbot, session_id, details, summary]
        # 先展示 AI 的答复，再生成首轮摘要并更新左侧标题；额外模型延迟不挡住正文。
        for event in (
            send.click(chat_app.respond_with_stream, submit_inputs, submit_outputs),
            message.submit(chat_app.respond_with_stream, submit_inputs, submit_outputs),
        ):
            # 标题摘要可能再调用一次模型；按钮要先显示，不能被标题生成延迟挡住。
            event.then(chat_app.pending_button_updates, inputs=[details],
                       outputs=[pending_hint, confirm_pending, abandon_pending]).then(
                           chat_app.refresh_session_title,
                           inputs=[user_id, session_id, details], outputs=[sessions],
                       )

        for button, handler in (
            (confirm_pending, chat_app.confirm_pending_button),
            (abandon_pending, chat_app.abandon_pending_button),
        ):
            button.click(
                handler, inputs=[chatbot, user_id, session_id, details], outputs=submit_outputs,
            ).then(chat_app.pending_button_updates, inputs=[details],
                   outputs=[pending_hint, confirm_pending, abandon_pending]).then(
                       chat_app.refresh_session_title,
                       inputs=[user_id, session_id, details], outputs=[sessions],
                   )

        load_outputs = [sessions, session_id, chatbot, details, summary, message]
        for event in (
            demo.load(chat_app.load_user_sessions, inputs=[user_id], outputs=load_outputs),
            user_id.change(chat_app.load_user_sessions, inputs=[user_id], outputs=load_outputs),
            refresh.click(chat_app.load_user_sessions, inputs=[user_id], outputs=load_outputs),
        ):
            event.then(chat_app.pending_button_updates, inputs=[details],
                       outputs=[pending_hint, confirm_pending, abandon_pending]).then(
                           chat_app.summarize_loaded_session,
                           inputs=[user_id, session_id], outputs=[sessions],
                       )
        clear.click(chat_app.new_session, inputs=[user_id], outputs=load_outputs).then(
            chat_app.pending_button_updates, inputs=[details],
            outputs=[pending_hint, confirm_pending, abandon_pending],
        )
        # Chatbot 自带的清空图标也视为“新对话”，不能在旧 checkpoint 上继续发消息。
        chatbot.clear(chat_app.new_session, inputs=[user_id], outputs=load_outputs).then(
            chat_app.pending_button_updates, inputs=[details],
            outputs=[pending_hint, confirm_pending, abandon_pending],
        )
        sessions.input(chat_app.select_session, inputs=[user_id, sessions],
                       outputs=load_outputs).then(chat_app.pending_button_updates,
                                                  inputs=[details],
                                                  outputs=[pending_hint, confirm_pending, abandon_pending],
                                                  ).then(chat_app.summarize_loaded_session,
                                                         inputs=[user_id, session_id], outputs=[sessions])
        restore.click(chat_app.restore_with_sidebar, inputs=[user_id, session_id],
                      outputs=[sessions, chatbot, details, summary]).then(
                          chat_app.pending_button_updates, inputs=[details],
                          outputs=[pending_hint, confirm_pending, abandon_pending],
                      ).then(chat_app.summarize_loaded_session,
                             inputs=[user_id, session_id], outputs=[sessions])
        session_id.input(chat_app.manual_session_input, inputs=[user_id, session_id],
                         outputs=[sessions, chatbot, details, summary]).then(
                             chat_app.pending_button_updates, inputs=[details],
                             outputs=[pending_hint, confirm_pending, abandon_pending],
                         )
    return demo


def main() -> None:
    """启动 Gradio 聊天演示页面。"""
    print("*"*50)
    build_ui().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7862")),
    )


if __name__ == "__main__":
    main()
