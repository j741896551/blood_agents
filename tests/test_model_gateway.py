"""故障注入测试，不调用真实供应商，也不代表线上可用率。"""

import json
import time
from threading import Event, Lock, Thread

import httpx
import pytest
from blood_agents.graph import build_graph
from blood_agents.model_gateway import GatewayFailure, ModelGateway, ModelProvider
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


class Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=json.dumps(response))


class StreamingModel:
    def __init__(self, *chunks):
        self.chunks = chunks
        self.stream_requests = []

    def stream(self, messages):
        self.stream_requests.append(messages)
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield AIMessageChunk(content=chunk)


def pair(primary, backup, **options):
    gateway = ModelGateway([ModelProvider("primary", primary), ModelProvider("backup", backup)], **options)
    return gateway, IntentRouter(gateway)


def test_primary_success():
    primary, backup = Model({"intent": "points"}), Model()
    _, router = pair(primary, backup)
    result, info = router.navigate("积分")
    assert result.intent == "points"
    assert info["provider"] == "primary"
    assert not backup.requests


@pytest.mark.parametrize("error", [TimeoutError("secret"), ConnectionError("secret")])
def test_switch_same_input_and_redacted_trace(error):
    primary, backup = Model(error), Model({"intent": "appointment"})
    _, router = pair(primary, backup)
    _, info = router.navigate("预约", active_agent="appointment")
    assert info["provider"] == "backup"
    assert primary.requests[0] == backup.requests[0]
    assert "secret" not in json.dumps(info)


@pytest.mark.parametrize("status,expected", [(429, "backup"), (503, "backup"), (401, None), (400, None)])
def test_real_sdk_http_errors(status, expected):
    def handle(request):
        return httpx.Response(status, json={"error": {"message": "secret", "type": "mock"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        primary = ChatOpenAI(model="demo", api_key="fake", base_url="http://mock/v1",
                             http_client=client, max_retries=0)
        backup = Model({"intent": "points"})
        _, router = pair(primary, backup)
        _, info = router.navigate("积分")
    assert info["provider"] == expected
    assert len(backup.requests) == (1 if expected else 0)
    assert "secret" not in json.dumps(info)


def test_circuit_skip_recovery_and_failed_probe():
    primary = Model(TimeoutError(), ConnectionError(), {"intent": "points"})
    backup = Model(*[{"intent": "points"}] * 3)
    gateway, router = pair(primary, backup, failure_threshold=1)
    router.navigate("积分")
    _, info = router.navigate("积分")
    assert info["attempts"][0]["status"] == "circuit_open"
    gateway.providers[0].open_until = 0  # 模拟冷却到期，无需等待真实 30 秒。
    assert router.navigate("积分")[1]["provider"] == "backup"
    assert gateway.providers[0].open_until > time.monotonic()
    gateway.providers[0].open_until = 0
    assert router.navigate("积分")[1]["provider"] == "primary"
    assert gateway.providers[0].failures == 0


def test_all_fail_retains_input_slots_and_does_not_confirm():
    slots = {"site": "市中心献血屋", "status": "awaiting_confirmation"}
    _, router = pair(Model(TimeoutError()), Model(ConnectionError()))
    parsed, info = router.interpret_appointment("好的", slots=slots)
    assert slots == {"site": "市中心献血屋", "status": "awaiting_confirmation"}
    assert parsed.action == "collect"
    assert info["mode"] == "rules_fallback"
    assert len(info["attempts"]) == 2


def test_invalid_contract_does_not_switch():
    backup = Model()
    _, router = pair(Model({"intent": "points", "user_id": "forged"}), backup)
    assert router.navigate("积分")[1]["mode"] == "rules_fallback"


def test_stream_emits_chunks_then_validates_complete_answer():
    model = StreamingModel("你好", "，欢迎使用。")
    gateway = ModelGateway([ModelProvider("primary", model)])
    emitted = []
    answer, info = gateway.stream([], lambda text: text if text.endswith("。") else
                                  (_ for _ in ()).throw(TypeError("unfinished")), emitted.append)
    assert emitted == ["你好", "，欢迎使用。"]
    assert answer == "你好，欢迎使用。"
    assert info["provider"] == "primary"


def test_stream_fails_over_only_before_first_visible_chunk():
    primary = StreamingModel(ConnectionError("offline"))
    backup = StreamingModel("备用回复")
    gateway = ModelGateway([ModelProvider("primary", primary), ModelProvider("backup", backup)])
    emitted = []
    answer, info = gateway.stream([], lambda text: text, emitted.append)
    assert answer == "备用回复"
    assert emitted == ["备用回复"]
    assert info["provider"] == "backup"

    primary = StreamingModel("已显示片段", ConnectionError("offline"))
    backup = StreamingModel("不应拼接")
    gateway = ModelGateway([ModelProvider("primary", primary), ModelProvider("backup", backup)])
    emitted = []
    with pytest.raises(GatewayFailure):
        gateway.stream([], lambda text: text, emitted.append)
    assert emitted == ["已显示片段"]
    assert not backup.stream_requests


def test_invalid_output_reports_safe_error_class_without_raw_response():
    class BadOutput:
        def invoke(self, messages):
            return AIMessage(content="secret invalid json")

    backup = Model()
    _, router = pair(BadOutput(), backup)
    parsed, info = router.navigate("你好")
    assert parsed.intent == "general"
    assert info["mode"] == "rules_fallback"
    assert info["error_type"] == "invalid_model_output"
    assert info["attempts"][0]["exception_class"] == "ValidationError"
    assert "secret" not in json.dumps(info)
    assert not backup.requests


def test_unexpected_model_exception_is_not_mislabeled_as_invalid_output():
    _, router = pair(Model(RuntimeError("secret")), Model())
    _, info = router.navigate("你好")
    assert info["error_type"] == "model_error"
    assert info["attempts"][0]["exception_class"] == "RuntimeError"
    assert "secret" not in json.dumps(info)


def test_bounded_timeout_single_inflight_probe():
    entered, release = Event(), Event()

    class BlockingModel:
        def invoke(self, messages):
            entered.set()
            release.wait(2)
            return AIMessage(content='{"intent":"points"}')

    gateway, router = pair(BlockingModel(), Model({"intent": "points"}, {"intent": "points"}))
    gateway.providers[0].timeout = 0.03
    gateway.total_timeout = 0.2
    try:
        started = time.monotonic()
        assert router.navigate("积分")[1]["provider"] == "backup"
        assert time.monotonic() - started < 0.5
        assert entered.is_set()
        assert router.navigate("积分")[1]["attempts"][0]["status"] == "busy"
    finally:
        release.set()


def test_timed_out_primary_keeps_slot_until_late_result_and_backup_wins(monkeypatch):
    """网关超时不等于底层请求已停止；迟到结果不能覆盖备用答复。"""
    entered, release = Event(), Event()
    late_seen = Event()
    late_status = []

    def record_event(name, **fields):
        if name == "model.attempt.late_result":
            late_status.append(fields["summary"]["status"])
            late_seen.set()

    monkeypatch.setattr("blood_agents.llm.model_gateway.trace_event", record_event)

    class LatePrimary:
        def invoke(self, messages):
            entered.set()
            release.wait(2)
            return AIMessage(content="late primary")

    class FastBackup:
        def __init__(self):
            self.answers = iter(("backup", "backup again"))

        def invoke(self, messages):
            return AIMessage(content=next(self.answers))

    backup = FastBackup()
    primary_provider = ModelProvider("primary", LatePrimary(), timeout=0.1)
    gateway = ModelGateway([
        primary_provider, ModelProvider("backup", backup, timeout=0.2),
    ], total_timeout=0.3, failure_threshold=10)
    try:
        answer, info = gateway.call([], lambda value: value)
        assert entered.is_set()
        assert answer == "backup"
        assert [attempt["status"] for attempt in info["attempts"]] == ["failed", "success"]
        assert info["attempts"][0]["error_type"] == "timeout"
        assert primary_provider.active == 1  # 真实请求未退出，不能提前释放并发名额。
        second, second_info = gateway.call([], lambda value: value)
        assert second == "backup again"
        assert second_info["attempts"][0]["status"] == "busy"
    finally:
        release.set()
    assert late_seen.wait(1)
    assert late_status == ["success_ignored"]
    for _ in range(100):
        if primary_provider.active == 0:
            break
        time.sleep(0.01)
    assert primary_provider.active == 0


def test_primary_timeout_and_backup_failure_do_not_return_late_answer():
    entered, release = Event(), Event()

    class LatePrimary:
        def invoke(self, messages):
            entered.set()
            release.wait(2)
            return AIMessage(content="late primary")

    primary_provider = ModelProvider("primary", LatePrimary(), timeout=0.1)
    gateway = ModelGateway([
        primary_provider, ModelProvider("backup", Model(ConnectionError("private detail"))),
    ], total_timeout=0.3, failure_threshold=10)
    try:
        with pytest.raises(GatewayFailure) as caught:
            gateway.call([], lambda value: value)
        assert entered.is_set()
        assert caught.value.info["mode"] == "rules_fallback"
        assert [item["error_type"] for item in caught.value.info["attempts"]] == [
            "timeout", "connection",
        ]
        assert "private detail" not in json.dumps(caught.value.info)
        assert primary_provider.active == 1
    finally:
        release.set()
    for _ in range(100):
        if primary_provider.active == 0:
            break
        time.sleep(0.01)
    assert primary_provider.active == 0


@pytest.mark.parametrize("first_chunk", [False, True])
def test_stream_timeout_switches_only_before_any_visible_chunk(first_chunk):
    entered, release = Event(), Event()

    class SlowPrimary:
        def stream(self, messages):
            entered.set()
            if first_chunk:
                yield AIMessageChunk(content="已显示草稿")
            release.wait(2)
            yield AIMessageChunk(content="迟到内容")

    primary_provider = ModelProvider("primary", SlowPrimary(), timeout=0.1)
    backup = StreamingModel("备用完整回复")
    gateway = ModelGateway([
        primary_provider, ModelProvider("backup", backup, timeout=0.2),
    ], total_timeout=0.3, failure_threshold=10)
    visible = []
    try:
        if first_chunk:
            with pytest.raises(GatewayFailure) as caught:
                gateway.stream([], lambda value: value, visible.append)
            assert caught.value.info["mode"] == "unavailable"
            assert visible == ["已显示草稿"]
            assert not backup.stream_requests
        else:
            answer, info = gateway.stream([], lambda value: value, visible.append)
            assert answer == "备用完整回复"
            assert info["provider"] == "backup"
            assert visible == ["备用完整回复"]
        assert entered.is_set()
        assert primary_provider.active == 1
    finally:
        release.set()
    for _ in range(100):
        if primary_provider.active == 0:
            break
        time.sleep(0.01)
    assert primary_provider.active == 0
    assert "迟到内容" not in "".join(visible)


def test_configured_provider_allows_two_parallel_requests_and_bounds_the_third():
    entered, release = Event(), Event()
    gate = Lock()
    counts = {"active": 0, "peak": 0}

    class BlockingModel:
        def invoke(self, messages):
            with gate:
                counts["active"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
                if counts["active"] == 2:
                    entered.set()
            release.wait(2)
            with gate:
                counts["active"] -= 1
            return AIMessage(content="ok")

    gateway = ModelGateway([ModelProvider("primary", BlockingModel(),
                                          max_concurrency=2)], total_timeout=2)
    results = []
    workers = [Thread(target=lambda: results.append(gateway.call([], lambda text: text)))
               for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(GatewayFailure) as exc:
            gateway.call([], lambda text: text)
        assert exc.value.info["attempts"][0]["status"] == "busy"
        assert counts["peak"] == 2
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
    assert len(results) == 2
    assert all(info["provider"] == "primary" for _, info in results)
    assert gateway.providers[0].active == 0


def test_invalid_provider_concurrency_rejected():
    for value in (0, -1, 65, True):
        with pytest.raises(ValueError, match="max_concurrency"):
            ModelProvider("primary", Model(), max_concurrency=value)


def test_total_budget_skips_backup():
    release = Event()

    class BlockingModel:
        def invoke(self, messages):
            release.wait(2)
            return AIMessage(content='{"intent":"points"}')

    backup = Model()
    _, router = pair(BlockingModel(), backup, total_timeout=0.02)
    try:
        _, info = router.navigate("积分")
        assert info["error_type"] == "budget_exhausted"
        assert not backup.requests
    finally:
        release.set()


def test_concurrent_half_open_has_only_one_probe():
    entered, release = Event(), Event()

    class Probe:
        def invoke(self, messages):
            entered.set()
            release.wait(2)
            return AIMessage(content='{"intent":"points"}')

    gateway, router = pair(Probe(), Model({"intent": "points"}))
    gateway.providers[0].failures = 2
    worker = Thread(target=lambda: router.navigate("积分"))
    worker.start()
    try:
        assert entered.wait(1)
        _, info = router.navigate("积分")
        assert info["provider"] == "backup"
        assert info["attempts"][0]["status"] == "busy"
    finally:
        release.set()
        worker.join(2)


def test_half_open_stays_single_probe_even_with_parallel_capacity():
    entered, release = Event(), Event()

    class Probe:
        def invoke(self, messages):
            entered.set()
            release.wait(2)
            return AIMessage(content="ok")

    primary = ModelProvider("primary", Probe(), max_concurrency=2)
    primary.failures = 2
    primary.open_until = time.monotonic() - 1
    gateway = ModelGateway([primary, ModelProvider("backup", Model({"intent": "points"}))], total_timeout=2)
    worker = Thread(target=lambda: gateway.call([], lambda text: text))
    worker.start()
    try:
        assert entered.wait(1)
        # 仍有第 2 个名额，但恢复探测只能有 1 个；其余请求不得挤进主服务。
        assert primary.active == 1
        assert primary.probe_active is True
        assert primary.active < primary.max_concurrency
        _, info = gateway.call([], lambda text: text)
        assert info["provider"] == "backup"
        assert info["attempts"][0]["status"] == "busy"
    finally:
        release.set()
        worker.join(2)


def test_independent_environment_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "primary-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://primary.local/v1")
    monkeypatch.setenv("BACKUP_OPENAI_API_KEY", "backup-test")
    monkeypatch.setenv("BACKUP_OPENAI_BASE_URL", "http://backup.local/v1")
    monkeypatch.setenv("BACKUP_OPENAI_MODEL", "backup-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "40")
    monkeypatch.setenv("BACKUP_LLM_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("LLM_TOTAL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("BACKUP_LLM_MAX_CONCURRENCY", "2")
    router = IntentRouter.from_env()
    primary, backup = router.llm.providers
    assert str(primary.model.openai_api_base) == "http://primary.local/v1"
    assert str(backup.model.openai_api_base) == "http://backup.local/v1"
    assert backup.model.model_name == "backup-model"
    assert primary.model.max_retries == backup.model.max_retries == 0
    assert (primary.max_concurrency, backup.max_concurrency) == (3, 2)
    assert (primary.timeout, backup.timeout, router.llm.total_timeout) == (40, 20, 60)


def test_fastapi_chat_fails_over_from_primary_http_503_to_backup(monkeypatch):
    """用 HTTP Mock 走完整 /chat → 导航 → 回复链路，验证平台 API 主备接线。"""
    from blood_agents.conversation import ConversationGraph
    from fastapi.testclient import TestClient

    from blood_agents import api

    monkeypatch.setenv("BLOOD_AGENT_RISK_MODE", "passthrough")
    calls = {"primary": 0, "backup": 0}

    def primary_handler(request):
        calls["primary"] += 1
        return httpx.Response(503, json={
            "error": {"message": "simulated primary outage", "type": "server_error"},
        })

    def completion(content):
        return httpx.Response(200, json={
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 1,
            "model": "mock", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
        })

    def backup_handler(request):
        calls["backup"] += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        # 导航请求的输入契约是 context/message；回复节点输入的是结构化业务 facts。
        if "context" in user_payload and "message" in user_payload:
            return completion('{"intent":"general"}')
        return completion(
                "你好！我是血站智慧献血导航，可以帮你：\n\n"
                "- **献血预约**：预约、查询、取消和签到\n"
                "- **积分与礼品**：查询积分和兑换礼品\n"
                "- **献血记录**：查询历史记录\n"
                "- **献血知识**：回答有依据的知识问题"
        )

    primary_client = httpx.Client(transport=httpx.MockTransport(primary_handler))
    backup_client = httpx.Client(transport=httpx.MockTransport(backup_handler))
    primary = ChatOpenAI(model="primary-demo", api_key="primary-test",
                         base_url="http://primary.mock/v1", http_client=primary_client,
                         max_retries=0, timeout=1)
    backup = ChatOpenAI(model="backup-demo", api_key="backup-test",
                        base_url="http://backup.mock/v1", http_client=backup_client,
                        max_retries=0, timeout=1)
    gateway = ModelGateway([
        ModelProvider("primary", primary, timeout=1),
        ModelProvider("backup", backup, timeout=1),
    ], total_timeout=4, failure_threshold=10)
    router = IntentRouter(gateway, response_llm=gateway, tool_llm=gateway,
                          risk_llm=False, evidence_llm=False)
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(repo, router, checkpointer=MemorySaver()))
    monkeypatch.setattr(api, "graph", graph)

    try:
        with TestClient(api.app) as client:
            response = client.post("/chat", json={
                "user_id": "U1001", "session_id": "api-failover", "message": "你好",
            })
        assert response.status_code == 200
        payload = response.json()
        assert "你好" in payload["answer"]
        assert payload["model_info"]["provider"] == "backup"
        assert payload["response_info"]["provider"] == "backup"
        assert payload["response_info"]["mode"] == "llm"
        assert calls == {"primary": 2, "backup": 2}
    finally:
        primary_client.close()
        backup_client.close()


def test_graph_fallback_preserves_preview_then_failover_submits_only_once():
    repo = BloodRepository()
    repo.seed_demo_data()

    class CountingAPI(MockLegacyBloodAPI):
        def __init__(self, repository):
            super().__init__(repository)
            self.submissions = 0

        def submit_appointment(self, payload, idempotency_key):
            self.submissions += 1
            return super().submit_appointment(payload, idempotency_key)

    api = CountingAPI(repo)
    day = api.site_slots[0]["slots"][0]["date"]
    primary = Model(
        {"intent": "appointment"},
        {"slots": {"date": day, "site": "市中心献血屋", "time_slot": "上午"}},
        TimeoutError(), TimeoutError(), TimeoutError(), TimeoutError(),
    )
    backup = Model(ConnectionError(), ConnectionError(), {"intent": "appointment"}, {"action": "confirm"})
    _, router = pair(primary, backup, failure_threshold=10)
    graph = build_graph(repo, router, legacy_api=api, checkpointer=MemorySaver())

    def ask(text):
        return graph.invoke({"user_id": "U1001", "messages": [HumanMessage(content=text)]},
                            config={"configurable": {"thread_id": "failover-test"}})

    preview = ask("预约")
    fallback = ask("好的")
    assert fallback["slots"] == preview["slots"]
    assert fallback["model_info"]["mode"] == "rules_fallback"
    assert api.submissions == 0
    blocked = ask("请提交刚才的预约")
    assert blocked["result"].get("success") is not True
    assert blocked["model_info"]["provider"] == "backup"
    assert api.submissions == 0  # 备用理解可用也不能补造上一轮未送达的预览。


def test_refusal_does_not_switch():
    class Refusal:
        def invoke(self, messages):
            return AIMessage(content="", additional_kwargs={"refusal": "mock refusal"})

    backup = Model()
    _, router = pair(Refusal(), backup)
    assert router.navigate("积分")[1]["mode"] == "rules_fallback"
    assert not backup.requests
