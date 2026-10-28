"""本地业务集成用例是可选的；纯 Agent 仓库不包含业务替身。"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """缺少本地业务插件时，不收集依赖它的集成测试模块。"""
    if find_spec("local_business") is not None:
        return None
    path = Path(str(collection_path))
    if path.suffix == ".py" and "local_business" in path.read_text(encoding="utf-8"):
        return True
    return None


def pytest_report_header(config) -> str | None:
    """在测试输出中明确说明本地业务集成用例是否可运行。"""
    if find_spec("local_business") is None:
        return "本地业务插件未安装：相关集成测试未收集，仅运行纯 Agent 测试。"
    return None
