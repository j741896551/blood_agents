"""显式稳定偏好的模型提取与确定性证据门禁；不提取业务状态或身份资料。"""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from blood_agents.llm.model_gateway import GatewayFailure, ModelGateway
from blood_agents.log.trace_logging import trace_event
from blood_agents.storage.privacy import sanitize_conversation_text

_SET_MARKER = re.compile(r"记住|以后|今后|长期|通常|习惯|偏好|更喜欢|优先|改成|换成")
_REVOKE_MARKER = re.compile(r"忘记|清除|删除.{0,6}偏好|取消.{0,6}偏好|不再偏好|不要再记")
_NEGATIVE = re.compile(r"不想|不去|不喜欢|不再")
_TEMPORARY = re.compile(r"这次|本次|今天|明天|后天|临时")
_SITE_HINT = re.compile(r"网点|献血点|献血屋|献血车|血站")
_TIME_HINT = re.compile(r"时间|时段|上午|下午")

EXTRACTION_PROMPT = (
    "你只提取用户本轮明确表达、适合跨会话复用的稳定偏好。只输出 schema JSON。"
    "仅支持 preferred_site（偏好献血网点）和 preferred_time_slot（偏好上午或下午）。"
    "用户只是在本次预约中选择网点/时段、询问可预约情况、引用历史记录、设置收货地址，"
    "或没有明确要求长期记住时，action=none。不得从预约工具结果、历史行为推断偏好。"
    "明确纠正旧偏好时 action=set 并取本轮新值；明确忘记某项偏好时 action=revoke，value=null。"
    "source_quote 必须是本轮用户原文中包含意图与值（撤销时含要撤销的偏好类别）的连续原句。"
    "值不得包含姓名、手机号、证件号、地址、积分、预约状态或订单资料。"
    "已有偏好仅用于识别用户要纠正/撤销哪一项，不得当作本轮新值。schema="
)


class PreferenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["none", "set", "revoke"]
    key: Literal["preferred_site", "preferred_time_slot"] | None = None
    value: str | None = None
    source_quote: str | None = None


class LongTermMemoryService:
    """模型只能提名候选；原话、类别和隐私约束由服务端复核。"""

    def __init__(self, store, llm=None):
        """初始化 LongTermMemoryService 的依赖与运行状态。"""
        self.store = store
        self.llm = llm

    @staticmethod
    def is_preference_query(text: str) -> bool:
        """查询已保存偏好走主导航答复，不误入预约流程或记忆写入。"""
        return bool(
            re.search(r"偏好|喜欢|习惯", text)
            and re.search(r"哪个|哪些|什么|记得|保存了什么|查询|查看|查一下|显示", text)
            and not re.search(r"记住|改成|换成|忘记|清除|删除", text)
            and not re.search(r"预约|兑换|献血记录|积分", text)
        )

    def capture(self, *, user_id: str, session_id: str, trace_id: str, text: str) -> dict:
        """从用户原话提取可保存的长期偏好事件。"""
        if self.is_preference_query(text):
            return {"status": "not_applicable"}
        if not (_SET_MARKER.search(text) or _REVOKE_MARKER.search(text)):
            return {"status": "not_applicable"}
        if not (_SITE_HINT.search(text) or _TIME_HINT.search(text)):
            return {"status": "not_applicable"}
        # 含常见个人资料的整句不送去新增的记忆提取阶段，也不存来源原文。
        if sanitize_conversation_text(text) != text:
            return {"status": "skipped_sensitive"}
        if not self.llm:
            return {"status": "model_unavailable"}
        try:
            existing = self.store.list_active_preferences(user_id)
            messages = [
                SystemMessage(content=EXTRACTION_PROMPT + json.dumps(
                    PreferenceDecision.model_json_schema(), ensure_ascii=False,
                )),
                HumanMessage(content=json.dumps({
                    "message": text,
                    "existing_preferences": [
                        {"key": item["preference_key"], "value": item["value"]}
                        for item in existing
                    ],
                }, ensure_ascii=False)),
            ]
            trace_event("memory.extract.request", summary={"stage": "long_term_memory"},
                        full={"model_input": [message.content for message in messages]})
            if isinstance(self.llm, ModelGateway):
                decision, info = self.llm.call(
                    messages, lambda content: PreferenceDecision.model_validate_json(content),
                    stage="long_term_memory",
                )
            else:
                output = self.llm.invoke(messages)
                decision = PreferenceDecision.model_validate_json(output.content)
                info = {"mode": "llm", "provider": "direct"}
            trace_event("memory.extract.result", summary={
                "action": decision.action, "key": decision.key, "provider": info.get("provider"),
            }, full={"decision": decision.model_dump()})
            return self._apply(decision, text, existing, user_id, session_id, trace_id)
        except GatewayFailure as exc:
            return {"status": "failed", "error_type": exc.info.get("error_type", "gateway_failure")}
        except Exception as exc:  # noqa: BLE001 - 记忆失败不能重放业务图或生成虚假成功
            trace_event("memory.extract.failed", summary={"error_type": type(exc).__name__})
            return {"status": "failed", "error_type": type(exc).__name__}

    def _apply(self, decision, text, existing, user_id, session_id, trace_id) -> dict:
        """根据当前输入更新长期记忆中的偏好条目。"""
        if decision.action == "none":
            return {"status": "ignored"}
        key, quote, value = decision.key, decision.source_quote, decision.value
        if not key or not quote or quote not in text or len(quote) > 160:
            return {"status": "rejected_evidence"}
        if sanitize_conversation_text(quote) != quote:
            return {"status": "rejected_evidence"}
        if decision.action == "set":
            if not value or value not in quote or not _SET_MARKER.search(quote):
                return {"status": "rejected_evidence"}
            if _NEGATIVE.search(quote) and not re.search(r"改成|换成", quote):
                return {"status": "rejected_evidence"}
            if _TEMPORARY.search(quote):
                return {"status": "rejected_evidence"}
            if key == "preferred_time_slot" and value not in {"上午", "下午"}:
                return {"status": "rejected_evidence"}
            if key == "preferred_site" and (not 2 <= len(value) <= 80 or not _SITE_HINT.search(quote)):
                return {"status": "rejected_evidence"}
            if sanitize_conversation_text(value) != value:
                return {"status": "rejected_evidence"}
        else:
            if value is not None or not _REVOKE_MARKER.search(quote):
                return {"status": "rejected_evidence"}
            hint = _SITE_HINT if key == "preferred_site" else _TIME_HINT
            if not hint.search(quote):
                return {"status": "rejected_evidence"}
            if not any(item["preference_key"] == key for item in existing):
                return {"status": "not_found", "key": key}
        saved = self.store.save_preference_event(
            user_id=user_id, key=key, action=decision.action, value=value,
            source_session_id=session_id, source_trace_id=trace_id, source_quote=quote,
        )
        status = ("saved" if decision.action == "set" else "revoked") if saved else "unchanged"
        trace_event("memory.event", summary={"status": status, "key": key, "action": decision.action})
        return {"status": status, "key": key, "value": value if status == "saved" else None}
