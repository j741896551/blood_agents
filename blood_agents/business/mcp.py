from __future__ import annotations

from .tools import ToolRegistry


class MCPToolGateway:
    """MCP 边界适配层：远端传输可替换，本地调用保留同一工具契约。"""

    def __init__(self, registry: ToolRegistry) -> None:
        """初始化 MCPToolGateway 的依赖与运行状态。"""
        self.registry = registry

    def list_tools(self) -> list[dict]:
        """列出当前 MCP 服务可用的工具定义。"""
        return [
            {"name": name, "description": self.registry._tools[name].description}
            for name in self.registry.names()
        ]

    def call_tool(self, name: str, arguments: dict) -> object:
        """按名称调用 MCP 服务提供的工具。"""
        return self.registry.invoke(name, **arguments)
