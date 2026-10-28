"""智慧献血多智能体包；旧模块路径集中映射到新的功能目录。

历史测试和外部示例曾导入 ``blood_agents.graph`` 等薄转发模块。目录重组后
在此统一登记旧路径，避免根目录继续堆放十余个无业务逻辑的包装文件。
新代码应直接导入 ``orchestration``、``business`` 等真实实现路径。
"""

from importlib import import_module
from sys import modules

from .orchestration.graph import build_graph

_LEGACY_MODULES = {
    "adapters": "knowledge.adapters",
    "agent_registry": "orchestration.agent_registry",
    "agents": "domains",
    "conversation": "orchestration.conversation",
    "graph": "orchestration.graph",
    "legacy_api": "business.legacy_api",
    "mcp": "business.mcp",
    "model_gateway": "llm.model_gateway",
    "models": "core.models",
    "ocr": "domains.ocr_service",
    "presentation": "ui.presentation",
    "rag": "knowledge.rag",
    "reply_composer": "llm.reply_composer",
    "repository": "storage.repository",
    "risk_agent": "llm.risk_agent",
    "router": "orchestration.router",
    "tool_planner": "llm.tool_planner",
    "tools": "business.tools",
    "trace_logging": "log.trace_logging",
}

for _old_name, _new_name in _LEGACY_MODULES.items():
    _module = import_module(f".{_new_name}", __name__)
    modules[f"{__name__}.{_old_name}"] = _module
    globals()[_old_name] = _module

del _old_name, _new_name, _module

__all__ = ["build_graph"]

