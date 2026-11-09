"""Mock 旧业务 HTTP 服务与客户端契约回归。"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from blood_agents.business.legacy_http import (
    LegacyAPIContractError,
    LegacyAPITransportError,
    LegacyHTTPClient,
)
from local_business.http_service import create_mock_legacy_app
from local_business.repository import BloodRepository


@pytest.fixture
def app_and_repo():
    repo = BloodRepository(":memory:")
    repo.seed_demo_data(include_appointment_samples=True)
    return create_mock_legacy_app(repo), repo


def test_mock_http_service_exposes_read_contracts_and_preserves_business_errors(app_and_repo):
    app, _ = app_and_repo
    client = TestClient(app)

    assert client.get("/health").json() == {
        "status": "ok", "service": "mock-legacy-blood-api",
    }
    points = client.get("/v1/users/U1001/points").json()
    assert points == {"success": True, "data": {"user_id": "U1001", "points": 1200}}

    records = client.get("/v1/users/U1001/donation-records").json()
    assert records["success"] is True
    assert records["data"]["records"][0]["donated_at"] == "2025-12-15"
    assert client.get("/v1/users/U404/points").json() == {
        "success": False,
        "code": "USER_NOT_FOUND",
        "message": "未找到用户积分账户",
    }

    rewards = client.get("/v1/rewards").json()
    assert rewards[0] == {
        "reward_id": "R001", "name": "纪念保温杯", "points_cost": 500, "stock": 20,
    }


def test_mock_http_service_reuses_existing_write_rules_without_bypassing_agent_confirmation(
    app_and_repo,
):
    app, _ = app_and_repo
    client = TestClient(app)
    saved = client.post("/v1/users/U1001/shipping-addresses", json={
        "recipient": "演示收货人", "mobile": "13800001111", "detail": "演示地址 1 号",
    }).json()
    assert saved["success"] is True
    addresses = client.get("/v1/users/U1001/shipping-addresses").json()
    assert any(item["recipient"] == "演示收货人"
               for item in addresses["data"]["addresses"])
    # 此服务只承载旧业务 API；预约/兑换是否允许调用仍由 Agent 的确认状态机控制。


def test_http_client_maps_legacy_queries_to_explicit_endpoints():
    seen: list[tuple[str, str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/points"):
            return httpx.Response(200, json={
                "success": True, "data": {"user_id": "U1001", "points": 1200},
            })
        if request.url.path.endswith("/appointment-history"):
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/rewards":
            return httpx.Response(200, json=[{
                "reward_id": "R001", "name": "纪念保温杯", "points_cost": 500, "stock": 20,
            }])
        return httpx.Response(404, json={"detail": "not found"})

    raw_client = httpx.Client(transport=httpx.MockTransport(handle), base_url="http://mock.test")
    client = LegacyHTTPClient("http://mock.test", http_client=raw_client)
    assert client.get_points("U1001")["data"]["points"] == 1200
    assert client.get_appointment_history("U1001") == []
    assert client.list_rewards()[0]["reward_id"] == "R001"
    assert seen == [
        ("GET", "/v1/users/U1001/points"),
        ("GET", "/v1/users/U1001/appointment-history"),
        ("GET", "/v1/rewards"),
    ]
    raw_client.close()


def test_http_client_distinguishes_transport_failure_from_empty_data():
    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic failure")

    client = LegacyHTTPClient(
        "http://mock.test",
        http_client=httpx.Client(transport=httpx.MockTransport(fail), base_url="http://mock.test"),
    )
    with pytest.raises(LegacyAPITransportError):
        client.get_donation_records("U1001")
    client.close()


def test_http_client_rejects_malformed_response_contract():
    raw_client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=[])),
        base_url="http://mock.test",
    )
    client = LegacyHTTPClient("http://mock.test", http_client=raw_client)
    with pytest.raises(LegacyAPIContractError):
        client.get_points("U1001")
    raw_client.close()


def test_http_client_converts_mock_redeem_validation_to_original_value_error():
    raw_client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(
            422, json={"detail": "积分不足"},
        )),
        base_url="http://mock.test",
    )
    client = LegacyHTTPClient("http://mock.test", http_client=raw_client)
    with pytest.raises(ValueError, match="积分不足"):
        client.redeem_reward("U1001", "R001", "demo-idempotency", "A0001")
    raw_client.close()


def test_graph_business_tools_use_injected_http_adapter():
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from blood_agents.orchestration.graph import build_graph
    from blood_agents.orchestration.router import IntentRouter

    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/points"):
            return httpx.Response(200, json={
                "success": True, "data": {"user_id": "U1001", "points": 1200},
            })
        if request.url.path == "/v1/rewards":
            return httpx.Response(200, json=[{
                "reward_id": "R001", "name": "纪念保温杯", "points_cost": 500, "stock": 20,
            }])
        return httpx.Response(404, json={"detail": "unexpected endpoint"})

    raw_client = httpx.Client(transport=httpx.MockTransport(handle), base_url="http://mock.test")
    http_api = LegacyHTTPClient("http://mock.test", http_client=raw_client)
    repo = BloodRepository(":memory:")
    repo.seed_demo_data()
    graph = build_graph(repo, IntentRouter(), legacy_api=http_api, checkpointer=MemorySaver())
    state = graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content="查询我的积分")]},
        {"configurable": {"thread_id": "http-query"}},
    )

    assert state["result"]["points"] == 1200
    assert state["result"]["rewards"][0]["reward_id"] == "R001"
    assert seen == ["/v1/users/U1001/points", "/v1/rewards"]
    assert [entry["backend"] for entry in state["tool_trace"]] == [
        "legacy_http", "legacy_http",
    ]
    raw_client.close()


def test_configured_business_transport_selects_http_adapter(tmp_path, monkeypatch):
    from blood_agents.orchestration.graph import _legacy_api_from_env

    monkeypatch.setenv("BLOOD_AGENT_BUSINESS_TRANSPORT", "http")
    monkeypatch.setenv("BLOOD_AGENT_LEGACY_API_URL", "http://127.0.0.1:8123")
    api = _legacy_api_from_env(BloodRepository(str(tmp_path / "agent.db")))
    assert isinstance(api, LegacyHTTPClient)
    assert api.transport_name == "legacy_http"
    api.close()
