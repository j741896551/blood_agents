"""MCP → HTTP → Mock 旧业务服务：协议、图接线和失败边界。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.business.legacy_http import LegacyHTTPClient
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.http_service import create_mock_legacy_app
from local_business.repository import BloodRepository

mcp = pytest.importorskip("mcp")


def _http_bridge(repo: BloodRepository):
    """不绑定端口的 HTTP 协议桥；MCP Server 只能通过 LegacyHTTPClient 读数据。"""
    app_client = TestClient(create_mock_legacy_app(repo))
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        response = app_client.request(request.method, request.url.path)
        return httpx.Response(response.status_code, content=response.content,
                              headers={"content-type": "application/json"})

    raw = httpx.Client(transport=httpx.MockTransport(handle), base_url="http://mock.test")
    return LegacyHTTPClient("http://mock.test", http_client=raw), raw, seen


def test_mcp_protocol_reads_same_http_service_and_preserves_user_isolation():
    from blood_agents.business.mcp_server import build_mcp_server

    repo = BloodRepository(":memory:")
    repo.seed_demo_data()
    http_api, raw, seen = _http_bridge(repo)

    async def check():
        async with mcp.Client(build_mcp_server(http_api)) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "blood_list_rewards", "blood_get_points", "blood_get_donation_records",
            }
            assert all(tool.annotations.read_only_hint for tool in listed.tools)
            assert (await client.call_tool("submit_appointment", {})).is_error is True

            rewards = await client.call_tool("blood_list_rewards", {})
            assert rewards.structured_content["rewards"] == http_api.list_rewards()
            for user_id in ("U1001", "U1002"):
                points = await client.call_tool("blood_get_points", {"user_id": user_id})
                records = await client.call_tool(
                    "blood_get_donation_records", {"user_id": user_id},
                )
                assert points.structured_content["reply"] == http_api.get_points(user_id)
                assert records.structured_content["reply"] == http_api.get_donation_records(user_id)
            assert points.structured_content["reply"]["data"]["user_id"] == "U1002"
            missing = await client.call_tool("blood_get_points", {"user_id": "U404"})
            assert missing.structured_content["reply"]["code"] == "USER_NOT_FOUND"

    try:
        asyncio.run(check())
        assert "/v1/rewards" in seen
        assert "/v1/users/U1001/points" in seen
        assert "/v1/users/U1002/donation-records" in seen
    finally:
        raw.close()


def test_mcp_http_failure_is_tool_error_not_empty_result():
    from blood_agents.business.mcp_server import build_mcp_server

    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("synthetic timeout")

    raw = httpx.Client(transport=httpx.MockTransport(fail), base_url="http://mock.test")
    http_api = LegacyHTTPClient("http://mock.test", http_client=raw)

    async def check():
        async with mcp.Client(build_mcp_server(http_api)) as client:
            result = await client.call_tool("blood_get_points", {"user_id": "U1001"})
            assert result.is_error is True
            assert result.structured_content is None

    try:
        asyncio.run(check())
    finally:
        raw.close()


def test_stdio_client_passes_http_url_not_database_path_to_server(monkeypatch):
    from blood_agents.business.mcp_client import MCPBusinessQueryClient

    seen = {}

    class FakeClient:
        def __init__(self, server):
            seen["server"] = server

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="blood_get_points")])

        async def call_tool(self, name, arguments):
            seen["call"] = (name, arguments)
            return SimpleNamespace(is_error=False, structured_content={
                "reply": {"success": True, "data": {"user_id": "U1001", "points": 1200}},
            })

    monkeypatch.setattr(mcp, "Client", FakeClient)
    adapter = MCPBusinessQueryClient("http://127.0.0.1:8001", timeout=7)
    assert adapter.get_points("U1001")["data"]["points"] == 1200
    assert seen["call"] == ("blood_get_points", {"user_id": "U1001"})
    assert seen["server"].env["BLOOD_AGENT_LEGACY_API_URL"] == "http://127.0.0.1:8001"
    assert seen["server"].env["BLOOD_AGENT_LEGACY_API_TIMEOUT"] == "7"
    assert "BLOOD_AGENT_DB" not in seen["server"].env


def test_graph_does_not_treat_mcp_read_failure_as_empty_business_data():
    from local_business.api import MockLegacyBloodAPI

    class FailedQueries:
        def get_points(self, _user_id):
            raise RuntimeError("synthetic upstream unavailable")

        def get_donation_records(self, _user_id):
            raise RuntimeError("synthetic upstream unavailable")

        def list_rewards(self):
            raise AssertionError("积分查询失败后不应继续查询目录")

    repo = BloodRepository(":memory:")
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(), legacy_api=MockLegacyBloodAPI(repo),
        mcp_queries=FailedQueries(), checkpointer=MemorySaver(),
    ))
    for question, expected_status in (
        ("查询我的积分", "points_query_failed"),
        ("查询我的献血记录", "records_query_failed"),
    ):
        state = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": expected_status}},
        )
        assert state["result"]["status"] == expected_status
        assert state["result"]["success"] is False
        assert state["tool_trace"][0]["status"] == "error"
        assert state["tool_trace"][0]["backend"] == "mcp_stdio"
        assert "records" not in state["result"]
        assert "points" not in state["result"]


def test_mcp_query_client_enters_graph_tools_and_tracks_backend(monkeypatch):
    from blood_agents.business.mcp_client import MCPBusinessQueryClient
    from blood_agents.business.mcp_server import build_mcp_server
    from blood_agents.orchestration.graph import build_conversational_graph

    repo = BloodRepository(":memory:")
    repo.seed_demo_data()
    http_api, raw, seen = _http_bridge(repo)
    server = build_mcp_server(http_api)

    async def in_memory_call(self, name, arguments):
        # 仅替换子进程传输，仍走真实 MCP tools/list + tools/call 协议。
        async with mcp.Client(server) as client:
            assert name in {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool(name, arguments)
            if result.is_error:
                raise RuntimeError("MCP 查询失败")
            return result.structured_content

    monkeypatch.setattr(MCPBusinessQueryClient, "_call", in_memory_call)
    mcp_queries = MCPBusinessQueryClient("http://mock.test")
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(), legacy_api=http_api, mcp_queries=mcp_queries,
        checkpointer=MemorySaver(),
    ))
    try:
        state = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="查询我的积分")]},
            {"configurable": {"thread_id": "mcp-points"}},
        )
        assert state["intent"] == "points"
        assert [item["tool"] for item in state["tool_trace"]] == ["query_points", "list_rewards"]
        assert [item["backend"] for item in state["tool_trace"]] == ["mcp_stdio"] * 2
        assert state["result"]["points"] == 1200
        assert state["result"]["rewards"][0]["reward_id"] == "R001"
        assert state["response_info"]["mode"] == "unavailable"  # 无模型不拼业务答复
        assert "/v1/users/U1001/points" in seen

        record_state = graph.invoke(
            {"user_id": "U1002", "messages": [HumanMessage(content="查询我的献血记录")]},
            {"configurable": {"thread_id": "mcp-records"}},
        )
        assert record_state["intent"] == "records"
        assert record_state["tool_trace"][0]["tool"] == "query_records"
        assert record_state["tool_trace"][0]["backend"] == "mcp_stdio"
        assert record_state["result"]["status"] in {"records_found", "no_records"}
        assert "/v1/users/U1002/donation-records" in seen

        monkeypatch.setenv("BLOOD_AGENT_BUSINESS_TRANSPORT", "mcp")
        monkeypatch.setenv("BLOOD_AGENT_LEGACY_API_URL", "http://mock.test")
        configured = build_conversational_graph(repo)
        configured_state = configured.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="查询我的积分")]},
            {"configurable": {"thread_id": "mcp-configured"}},
        )
        assert [item["backend"] for item in configured_state["tool_trace"]] == [
            "mcp_stdio", "mcp_stdio",
        ]
    finally:
        raw.close()
