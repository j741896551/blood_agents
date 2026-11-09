"""标准 MCP stdio Server：只做旧业务 HTTP 服务的只读协议适配。"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field

from .legacy_http import LegacyHTTPClient


class RewardEntry(BaseModel):
    reward_id: str
    name: str
    points_cost: int
    stock: int | None


class RewardCatalogResult(BaseModel):
    rewards: list[RewardEntry]


class BusinessReplyResult(BaseModel):
    """保留旧接口的 success/code/data，不能把业务失败改成空查询。"""

    reply: dict[str, Any]


def build_mcp_server(legacy_http: LegacyHTTPClient):
    """MCP 工具只转发 HTTP；不持有业务仓储，也不接收模型生成的身份。"""
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("blood_station_mcp")
    readonly = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @server.tool(name="blood_list_rewards", title="查询礼品目录", annotations=readonly)
    async def blood_list_rewards() -> RewardCatalogResult:
        """从外部业务 HTTP 服务查询礼品目录；不执行兑换。"""
        rows = await asyncio.to_thread(legacy_http.list_rewards)
        return RewardCatalogResult(rewards=[RewardEntry.model_validate(row) for row in rows])

    @server.tool(name="blood_get_points", title="查询用户积分", annotations=readonly)
    async def blood_get_points(
        user_id: str = Field(min_length=1, max_length=100, description="入口认证上下文中的用户 ID"),
    ) -> BusinessReplyResult:
        """按入口传来的 user_id 查询积分；返回旧服务原始成功或失败回执。"""
        reply = await asyncio.to_thread(legacy_http.get_points, user_id)
        return BusinessReplyResult(reply=reply)

    @server.tool(name="blood_get_donation_records", title="查询用户献血记录", annotations=readonly)
    async def blood_get_donation_records(
        user_id: str = Field(min_length=1, max_length=100, description="入口认证上下文中的用户 ID"),
    ) -> BusinessReplyResult:
        """按入口传来的 user_id 查询献血记录；空记录和查询失败分别保留。"""
        reply = await asyncio.to_thread(legacy_http.get_donation_records, user_id)
        return BusinessReplyResult(reply=reply)

    return server


def main() -> None:
    """启动 stdio MCP 服务，把工具请求转发给外部业务 HTTP 服务。"""
    # stdio 由 Agent 自动拉起；业务库只属于独立 HTTP 服务，不传 DB 路径。
    base_url = os.getenv("BLOOD_AGENT_LEGACY_API_URL", "http://127.0.0.1:8001")
    timeout = float(os.getenv("BLOOD_AGENT_LEGACY_API_TIMEOUT", "10"))
    client = LegacyHTTPClient(base_url, timeout=timeout)
    try:
        build_mcp_server(client).run(transport="stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
