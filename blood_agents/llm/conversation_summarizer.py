"""为会话列表生成标题和摘要；结果只作展示元数据，不参与 Agent 决策。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from blood_agents.log.trace_logging import trace_event
from blood_agents.storage.privacy import (
    REDACTED_MESSAGE,
    sanitize_conversation_text,
)

from .model_gateway import GatewayFailure, ModelGateway


class ConversationDigest(BaseModel):
    title: str = Field(min_length=4, max_length=28)
    summary: str = Field(min_length=8, max_length=240)


_PROMPT = (
    "你只负责给已完成的聊天会话生成展示元数据。根据用户与助手的实际对话，"
    "概括这段会话的主题与目前进展；标题用 4~28 个中文字符，摘要用 8~240 个字符。"
    "不要直接复制首句作为标题，不要推测未发生的操作或业务结果。"
    "不要包含姓名、电话、证件号、住址或其他个人资料。"
    "若对话只有问候或服务不可用，就客观描述，不能编造业务主题。"
    "只输出符合 schema 的 JSON，不要附加解释。schema="
)


def _parse_digest(content: str) -> ConversationDigest:
    """解析并校验模型生成的会话标题与摘要。"""
    if not isinstance(content, str):
        raise TypeError("conversation digest must be text")
    fenced = re.fullmatch(r"```(?:json)?\s*\n(?P<body>.*?)\n```", content.strip(),
                          re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced.group("body")
    digest = ConversationDigest.model_validate_json(content)
    title = " ".join(digest.title.split()).strip("《》“”\"'：:。.!！?？")
    summary = " ".join(digest.summary.split())
    if not 4 <= len(title) <= 28 or not 8 <= len(summary) <= 240:
        raise ValueError("conversation digest length is invalid")
    if (sanitize_conversation_text(title) != title
            or sanitize_conversation_text(summary) != summary
            or REDACTED_MESSAGE in title + summary):
        raise ValueError("conversation digest contains private data")
    return ConversationDigest(title=title, summary=summary)


class ConversationSummarizer:
    def __init__(self, llm: object | None) -> None:
        """初始化 ConversationSummarizer 的依赖与运行状态。"""
        self.llm = llm

    def summarize(self, turns: list[dict]) -> tuple[ConversationDigest | None, dict]:
        """只用仓储已脱敏的轮次；模型失败不影响聊天回复或业务操作。"""
        if not turns:
            return None, {"mode": "skipped", "reason": "empty_session"}
        if self.llm is None:
            return None, {"mode": "unavailable", "reason": "model_not_configured"}
        # 新会话首轮只有一组问答。旧会话补标题时取近期轮次，避免无界送模。
        context = [{"user": row["user_message"], "assistant": row["assistant_message"]}
                   for row in turns[-12:]]
        messages = [
            SystemMessage(content=_PROMPT + json.dumps(
                ConversationDigest.model_json_schema(), ensure_ascii=False,
            )),
            HumanMessage(content=json.dumps({"turns": context}, ensure_ascii=False)),
        ]
        trace_event("model.request", summary={"stage": "conversation_digest",
                                              "turn_count": len(context)},
                    full={"model_input": [{"role": m.type, "content": m.content} for m in messages]})
        try:
            if isinstance(self.llm, ModelGateway):
                digest, info = self.llm.call(messages, _parse_digest, stage="conversation_digest")
            else:
                response = self.llm.invoke(messages)
                trace_event("model.response", summary={"stage": "conversation_digest",
                                                       "provider": "direct"},
                            full={"model_output": response.content})
                digest, info = _parse_digest(response.content), {"mode": "llm", "provider": "direct"}
        except GatewayFailure as exc:
            trace_event("conversation.digest.failed", summary={
                "error_type": exc.info.get("error_type"), "mode": "unavailable",
            })
            return None, {"mode": "unavailable", "error_type": exc.info.get("error_type")}
        except Exception as exc:  # noqa: BLE001 - metadata failure cannot break the user reply
            trace_event("conversation.digest.failed", summary={
                "error_type": type(exc).__name__, "mode": "unavailable",
            })
            return None, {"mode": "unavailable", "error_type": type(exc).__name__}
        first_question = re.sub(r"[\s，,。.!！?？:：]+", "", turns[0]["user_message"])
        title_text = re.sub(r"[\s，,。.!！?？:：]+", "", digest.title)
        if title_text == first_question:
            trace_event("conversation.digest.failed", summary={
                "error_type": "title_copied_first_message", "mode": "unavailable",
            })
            return None, {"mode": "unavailable", "error_type": "title_copied_first_message"}
        trace_event("conversation.digest.generated", summary={
            "stage": "conversation_digest", "mode": info.get("mode"), "provider": info.get("provider"),
        })
        return digest, info
