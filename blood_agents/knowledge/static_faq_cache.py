"""只缓存人工确认的、与个人及实时业务无关的完整标准问答。"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _normalize(text: str) -> str:
    """仅容忍空白和句末标点；不做模糊/关键词匹配，避免混合意图误命中。"""
    return unicodedata.normalize("NFKC", text).strip().rstrip("。？！?! ").replace(" ", "")


@dataclass(frozen=True)
class StaticAnswer:
    key: str
    version: str
    expires_on: date
    answer: str
    questions: tuple[str, ...]
    approved: bool
    review_basis: str


class StaticFaqCache:
    """条目由项目文件审核/版本化；歧义或无效配置一律回原图。"""

    def __init__(self, entries: tuple[StaticAnswer, ...] = (), *, available: bool = True):
        """初始化 StaticFaqCache 的依赖与运行状态。"""
        self.entries = entries
        self.available = available

    @classmethod
    def from_file(cls, path: Path | None = None) -> StaticFaqCache:
        """从配置文件加载经审核的静态问答缓存。"""
        path = path or Path(__file__).with_name("static_faq_cache.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1 or not isinstance(data.get("entries"), list):
                raise ValueError("unsupported cache catalog")
            entries = tuple(StaticAnswer(
                key=item["key"], version=item["version"],
                expires_on=date.fromisoformat(item["expires_on"]),
                answer=item["answer"], questions=tuple(item["questions"]),
                approved=item["approved"], review_basis=item["review_basis"],
            ) for item in data["entries"])
            if any(not entry.key or not entry.version or not entry.answer.strip()
                   or not entry.questions or not entry.review_basis
                   or not isinstance(entry.approved, bool)
                   or any(not question.strip() for question in entry.questions)
                   for entry in entries):
                raise ValueError("invalid cache entry")
            return cls(entries)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # 配置损坏不能让静态文案代替业务 Agent，也不能阻断原有模型/RAG 路径。
            return cls(available=False)

    def lookup(self, question: str, *, active_agent: str | None = None,
               today: date | None = None) -> tuple[StaticAnswer | None, str]:
        """仅在完整匹配且满足边界条件时读取静态问答。"""
        if active_agent:
            return None, "active_task"
        if not self.available:
            return None, "catalog_unavailable"
        if not isinstance(question, str) or not question.strip():
            return None, "empty_question"
        normalized = _normalize(question)
        if not normalized:
            return None, "empty_question"
        today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
        matches = [entry for entry in self.entries
                   if entry.approved and entry.expires_on > today
                   and normalized in {_normalize(alias) for alias in entry.questions}]
        if len(matches) != 1:
            return None, "ambiguous" if len(matches) > 1 else "no_unique_match"
        return matches[0], "hit"
