"""Agent 对话表的保守文本留存策略。

检测到常见身份、联系方式或住址线索时，整条消息不入库；这不是
通用 PII 识别器，真实用户场景仍需要专门的隐私方案和审核。
"""

from __future__ import annotations

import re

from blood_agents.log.trace_logging import mask_sensitive

REDACTED_MESSAGE = "[涉及个人资料，消息内容未保存]"

_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)1[3-9]\d\*{4}\d{4}(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{3}\*{11}[\dXx]{4}(?!\d)")
_EMAIL = re.compile(r"(?<![\w.])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
_LABELED_IDENTITY = re.compile(
    r"(?:姓名|收件人|收货人|联系人|献血人)\s*(?:[:：=]|是|为)\s*[^，,。；;\s]{1,30}"
)
_LABELED_ADDRESS = re.compile(
    r"(?:收货地址|详细地址|家庭住址|居住地址|住址|地址)\s*(?:[:：=]|是|为)\s*\S+"
)
_STREET_ADDRESS = re.compile(
    r"(?:省|市|区|县).{0,50}(?:路|街|巷|小区|大厦|公寓).{0,30}(?:号|栋|楼|室)"
)


def sanitize_conversation_text(value: str) -> str:
    """只保留未检测到常见个人资料的文本；命中时不存原文或局部掩码。"""
    if not isinstance(value, str):
        raise TypeError("conversation text must be a string")
    if any(pattern.search(value) for pattern in (
        _PHONE, _ID_CARD, _EMAIL, _LABELED_IDENTITY, _LABELED_ADDRESS, _STREET_ADDRESS,
    )):
        return REDACTED_MESSAGE
    # 继续沿用日志掩码作为兜底；启发式检测无法保证识别所有自由文本资料。
    return mask_sensitive(value)
