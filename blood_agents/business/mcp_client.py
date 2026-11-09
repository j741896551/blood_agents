"""Agent 侧 MCP 查询 adapter；stdio Server 再调用同一旧业务 HTTP 服务。"""

from __future__ import annotations

import asyncio
import sys
from typing import Any


class MCPBusinessQueryClient:
    """只暴露经 MCP 协议发现的只读查询，写操作仍由 HTTP adapter 执行。"""

    transport_name = "mcp_stdio"

    def __init__(self, base_url: str, *, timeout: float = 10.0,
                 python_executable: str | None = None) -> None:
        """初始化 MCPBusinessQueryClient 的依赖与运行状态。"""
        if not base_url.strip():
            raise ValueError("MCP 业务查询需要旧服务 HTTP URL")
        self.base_url = base_url
        self.timeout = timeout
        self.python_executable = python_executable or sys.executable

    async def _call(self, name: str, arguments: dict[str, str]) -> dict[str, Any]:
        """调用 MCP 工具并取得原始返回结果。"""
        from mcp import Client, StdioServerParameters

        server = StdioServerParameters(
            command=self.python_executable,
            args=["-m", "blood_agents.business.mcp_server"],
            env={
                "BLOOD_AGENT_LEGACY_API_URL": self.base_url,
                "BLOOD_AGENT_LEGACY_API_TIMEOUT": str(self.timeout),
                "BLOOD_AGENT_TRACE_MODE": "off",
            },
        )
        async with Client(server) as client:
            # 标准 tools/list + tools/call；失败不得回落到本地 SQLite 伪装成功。
            names = {tool.name for tool in (await client.list_tools()).tools}
            if name not in names:
                raise RuntimeError(f"MCP Server 未提供 {name}")
            result = await client.call_tool(name, arguments)
            if result.is_error or not isinstance(result.structured_content, dict):
                raise RuntimeError(f"MCP 查询 {name} 失败或回执格式异常")
            return result.structured_content

    @staticmethod
    def _reply(content: dict[str, Any], name: str) -> dict[str, Any]:
        """把 MCP 工具返回值整理为业务响应结构。"""
        reply = content.get("reply")
        if not isinstance(reply, dict) or not isinstance(reply.get("success"), bool):
            raise TypeError(f"MCP 查询 {name} 的旧业务回执无效")
        return reply

    def list_rewards(self) -> list[dict[str, Any]]:
        """列出当前可供查询的礼品目录。"""
        content = asyncio.run(self._call("blood_list_rewards", {}))
        rewards = content.get("rewards")
        if not isinstance(rewards, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("reward_id"), str)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("points_cost"), int)
            or not (isinstance(item.get("stock"), int) or item.get("stock") is None)
            for item in rewards
        ):
            raise RuntimeError("MCP 礼品目录字段不符合本地契约")
        return rewards

    def get_points(self, user_id: str) -> dict[str, Any]:
        """按用户查询当前积分余额。"""
        return self._reply(
            asyncio.run(self._call("blood_get_points", {"user_id": user_id})),
            "blood_get_points",
        )

    def get_donation_records(self, user_id: str) -> dict[str, Any]:
        """按用户查询已有献血记录。"""
        return self._reply(
            asyncio.run(self._call("blood_get_donation_records", {"user_id": user_id})),
            "blood_get_donation_records",
        )
