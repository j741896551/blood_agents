"""一轮对话的结构化日志；ContextVar 让并发会话各自保有追踪身份。"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    user_id: str
    session_id: str
    entrypoint: str


current_trace: ContextVar[TraceContext | None] = ContextVar("blood_agent_trace", default=None)
_logger = logging.getLogger("blood_agents.trace")
_configure_lock = Lock()
_configured = False

# 英文 event 保持稳定，便于机器筛选；中文标签让本地排障时能直接读懂执行链。
EVENT_LABELS = {
    "turn.start": "收到用户问题",
    "session.lock_acquired": "取得会话锁",
    "navigation.rule_match": "导航规则匹配",
    "navigation.guardrail": "导航边界校验",
    "memory.short_term": "提取当前会话短期任务记忆",
    "memory.extract.request": "请求模型提取明确的长期偏好",
    "memory.extract.result": "长期偏好候选提取结果",
    "memory.extract.failed": "长期偏好提取失败",
    "memory.event": "长期偏好事件已处理",
    "memory.load_failed": "读取长期偏好失败",
    "cache.lookup": "标准问答缓存匹配结果",
    "model.request": "发送大模型输入",
    "model.attempt": "开始或跳过模型服务尝试",
    "model.response": "收到大模型原始输出",
    "model.stream.first_chunk": "收到流式回复首片",
    "model.attempt.finished": "模型服务尝试结束",
    "model.attempt.late_result": "超时请求最终返回",
    "understanding.result": "语义理解结果",
    "appointment.correction": "程序校正预约参数",
    "graph.dispatch": "主导航派发子智能体",
    "graph.node.start": "开始执行图节点",
    "graph.node.end": "图节点执行结束",
    "graph.node.error": "图节点执行失败",
    "appointment.understood": "预约子智能体理解结果",
    "appointment.preview_discarded": "预约确认预览未送达，撤销可执行确认状态",
    "points.understood": "积分子智能体理解结果",
    "points.action_guardrail": "待确认兑换的收货信息动作校正",
    "points.preview_discarded": "积分预览未送达，清除待确认操作",
    "agent.handoff": "智能体控制权交接",
    "tool.finished": "工具调用结束",
    "knowledge.search": "知识检索证据摘要",
    "evidence.check": "知识片段能否支持回答的核验结果",
    "reply.result": "回复表达结果",
    "reply.redacted": "回复中的个人信息已脱敏",
    "reply.quality_warning": "回复质量提示（未拦截）",
    "turn.end": "本轮回复完成",
    "turn.error": "本轮执行失败",
    "conversation.saved": "本轮对话已保存到数据库",
    "conversation.save_failed": "本轮对话保存失败",
    "conversation.digest.generated": "会话标题和摘要已由模型生成",
    "conversation.digest.failed": "会话标题和摘要生成失败",
    "conversation.digest.saved": "会话标题和摘要已写入数据库",
}
STAGE_LABELS = {"navigation": "主导航", "appointment": "预约子智能体",
                "points": "积分子智能体", "faq_evidence": "知识证据核验",
                "conversation_digest": "会话标题与摘要", "long_term_memory": "长期偏好提取"}


def mask_sensitive(value):
    """完整日志与页面统一遮盖用户输入中的手机号/合成身份证号。"""
    if isinstance(value, str):
        value = re.sub(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)", r"\1****\3", value)
        return re.sub(r"(?<!\d)(\d{3})(\d{11})([\dXx]{4})(?!\d)",
                      r"\1***********\3", value)
    if isinstance(value, dict):
        return {key: mask_sensitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_sensitive(item) for item in value]
    return value


def _configure_logger() -> None:
    """仅配置专用 logger，不改变 FastAPI/Gradio/应用其他日志的级别。"""
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        formatter = logging.Formatter("%(message)s")
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        _logger.addHandler(console)

        target = os.getenv("BLOOD_AGENT_TRACE_FILE", "").strip()
        if target:
            try:
                path = Path(target)
                path.parent.mkdir(parents=True, exist_ok=True)
                # 原始对话和模型提示词仅写入用户显式指定的文件，创建时限制权限。
                descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                os.chmod(path, 0o600)
                file_stream = os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)
                file_handler = logging.StreamHandler(file_stream)
                file_handler.setFormatter(formatter)
                _logger.addHandler(file_handler)
            except OSError as exc:
                _logger.warning(json.dumps({
                    "event": "trace.file_unavailable", "error_type": type(exc).__name__,
                }))
        _configured = True


def trace_event(event: str, *, summary: dict | None = None, full: dict | None = None) -> None:
    """本地调试默认写完整输入输出；部署时可改 summary 或 off。"""
    context = current_trace.get()
    mode = os.getenv("BLOOD_AGENT_TRACE_MODE", "full").strip().lower()
    if context is None or mode not in {"summary", "full"}:
        return
    _configure_logger()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "event_zh": EVENT_LABELS.get(event, event),
        "trace_mode": mode,
        "trace_id": context.trace_id,
        "session_id": context.session_id,
        "user_id": context.user_id,
        "entrypoint": context.entrypoint,
    }
    record.update(summary or {})
    if record.get("stage") in STAGE_LABELS:
        record["stage_zh"] = STAGE_LABELS[record["stage"]]
    if mode == "full":
        record.update(full or {})
    redacted = mask_sensitive(record)
    # trace_id 是服务端生成的随机关联键，数字序列偶然像手机号时也不能被替换，
    # 否则同一轮日志与 API/页面返回的 ID 不一致；其他字段仍按原规则脱敏。
    redacted["trace_id"] = context.trace_id
    _logger.info(json.dumps(redacted, ensure_ascii=False, default=str))
