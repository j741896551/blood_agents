"""验证整轮追踪能关联模型线程、规则校正和工具，同时限制摘要模式的原文输出。"""

import json
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from blood_agents.conversation import ConversationGraph
from blood_agents.graph import build_graph
from blood_agents.model_gateway import GatewayFailure, ModelGateway, ModelProvider
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents import trace_logging
from blood_agents.log.trace_logging import TraceContext, current_trace, trace_event
from blood_agents.log.trace_viewer import main as trace_viewer_main
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


class QueueModel:
    def __init__(self, *replies):
        self.replies = iter(replies)

    def invoke(self, _messages):
        return AIMessage(content=json.dumps(next(self.replies), ensure_ascii=False))


class EventCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append(json.loads(record.getMessage()))


def test_trace_id_is_not_phone_masked_but_user_text_still_is(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")
    context = TraceContext("abc13812345678def", "U1001", "session", "test")
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    token = current_trace.set(context)
    try:
        trace_event("turn.start", full={"user_question": "手机号 13812345678"})
    finally:
        current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)
    assert capture.events[0]["trace_id"] == context.trace_id
    assert capture.events[0]["user_question"] == "手机号 138****5678"


def test_stream_trace_records_first_chunk_and_counts_without_fragment_text(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")

    class StreamingModel:
        def stream(self, _messages):
            yield AIMessageChunk(content="第一段")
            yield AIMessageChunk(content="第二段")

    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    token = current_trace.set(TraceContext("stream-trace", "U1001", "stream-session", "test"))
    try:
        gateway = ModelGateway([ModelProvider("primary", StreamingModel())])
        answer, info = gateway.stream([], lambda text: text, lambda _chunk: None,
                                      stage="general_reply")
    finally:
        current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)

    assert answer == "第一段第二段"
    assert info["chunk_count"] == 2
    assert info["output_chars"] == len(answer)
    first = next(event for event in capture.events if event["event"] == "model.stream.first_chunk")
    assert first["trace_id"] == "stream-trace"
    assert first["stage"] == "general_reply"
    assert first["first_chunk_ms"] >= 0
    response = next(event for event in capture.events if event["event"] == "model.response")
    assert response["chunk_count"] == 2
    assert response["output_chars"] == len(answer)
    assert response["model_output"] == answer
    assert all("第一段" not in json.dumps(event, ensure_ascii=False)
               for event in capture.events if event["event"] != "model.response")


def conversation_with_model(*replies):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    gateway = ModelGateway([ModelProvider("primary", QueueModel(*replies))])
    graph = build_graph(repo, IntentRouter(gateway), legacy_api=api, checkpointer=MemorySaver())
    return ConversationGraph(graph), api


def test_full_trace_connects_question_model_raw_output_correction_and_tools(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")
    graph, api = conversation_with_model({"intent": "appointment"}, {})
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    try:
        state = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="我想明天下午来献血")]},
            {"configurable": {"thread_id": "trace-session"},
             "metadata": {"entrypoint": "test"}},
        )
    finally:
        trace_logging._logger.removeHandler(capture)

    events = capture.events
    assert events[0]["event"] == "turn.start"
    assert events[0]["user_question"] == "我想明天下午来献血"
    assert all(event["trace_id"] == state["trace_id"] for event in events)
    assert all(event["user_id"] == "U1001" and event["session_id"] == "trace-session"
               for event in events)
    requests = [event for event in events if event["event"] == "model.request"]
    responses = [event for event in events if event["event"] == "model.response"]
    assert [event["stage"] for event in requests] == ["navigation", "appointment"]
    assert "schema=" in requests[0]["model_input"][0]["content"]
    assert "我想明天下午来献血" in requests[1]["model_input"][1]["content"]
    assert [event["model_output"] for event in responses] == ['{"intent": "appointment"}', "{}"]
    correction = next(event for event in events if event["event"] == "appointment.correction")
    assert correction["before_correction"] == {}
    assert correction["literal_slots"] == {
        "date": api.site_slots[0]["slots"][0]["date"], "time_slot": "下午",
    }
    assert correction["after_correction"] == correction["literal_slots"]
    assert correction["corrected_slots"] == ["date", "time_slot"]
    assert [event["tool"] for event in events if event["event"] == "tool.finished"] == [
        "query_user_profile", "query_sites", "query_appointment_history",
    ]
    assert events[-1]["event"] == "turn.end"
    assert events[-1]["node_path"][-1] == "appointment_handoff"
    assert events[-1]["answer"] == state["messages"][-1].content


def test_default_trace_is_full_and_has_chinese_labels(monkeypatch):
    monkeypatch.delenv("BLOOD_AGENT_TRACE_MODE", raising=False)
    graph, _ = conversation_with_model({"intent": "points"})
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    try:
        graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="帮我查积分")]},
            {"configurable": {"thread_id": "default-full-session"}},
        )
    finally:
        trace_logging._logger.removeHandler(capture)
    request = next(event for event in capture.events if event["event"] == "model.request")
    response = next(event for event in capture.events if event["event"] == "model.response")
    assert request["trace_mode"] == "full"
    assert request["event_zh"] == "发送大模型输入"
    assert request["stage_zh"] == "主导航"
    assert "帮我查积分" in request["model_input"][1]["content"]
    assert response["event_zh"] == "收到大模型原始输出"
    assert response["model_output"] == '{"intent": "points"}'


def test_summary_trace_keeps_correlation_but_omits_raw_messages(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "summary")
    graph, _ = conversation_with_model(
        {"intent": "points"}, {"action": "query"},
        {"intent": "points"}, {"action": "query"},
    )
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    try:
        state = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="查询积分，私密备注FLAG")]},
            {"configurable": {"thread_id": "summary-session"}},
        )
        next_turn = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="再查一次积分")]},
            {"configurable": {"thread_id": "summary-session"}},
        )
    finally:
        trace_logging._logger.removeHandler(capture)

    serialized = json.dumps(capture.events, ensure_ascii=False)
    assert state["trace_id"] != next_turn["trace_id"]
    assert {event["trace_id"] for event in capture.events} == {
        state["trace_id"], next_turn["trace_id"],
    }
    assert "summary-session" in serialized and "U1001" in serialized
    assert "私密备注FLAG" not in serialized
    assert "model_input" not in serialized and "model_output" not in serialized


def test_failover_attempts_keep_one_trace_id_and_full_error_detail(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")

    class BrokenModel:
        def invoke(self, _messages):
            raise TimeoutError("private provider response")

    repo = BloodRepository()
    repo.seed_demo_data()
    gateway = ModelGateway([
        ModelProvider("primary", BrokenModel()),
        ModelProvider("backup", QueueModel({"intent": "points"}, {"action": "query"})),
    ])
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(gateway), legacy_api=MockLegacyBloodAPI(repo),
        checkpointer=MemorySaver(),
    ))
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    try:
        state = graph.invoke(
            {"user_id": "U1001", "messages": [HumanMessage(content="查询积分")]},
            {"configurable": {"thread_id": "failover-session"}},
        )
    finally:
        trace_logging._logger.removeHandler(capture)

    attempts = [event for event in capture.events if event["event"] == "model.attempt.finished"]
    assert [(event["provider"], event["status"]) for event in attempts] == [
        ("primary", "failed"), ("backup", "success"),
        ("primary", "failed"), ("backup", "success"),
    ]
    assert all(event["trace_id"] == state["trace_id"] for event in attempts)
    assert any(event["event"] == "model.response" and event["provider"] == "backup"
               for event in capture.events)
    assert attempts[0]["error_message"] == "private provider response"
    assert attempts[0]["cause_chain"] == [
        {"class": "TimeoutError", "message": "private provider response"},
    ]
    assert "error_message" not in state["model_info"]["navigation"]["attempts"][0]


def test_model_error_detail_redacts_keys_and_keeps_root_cause_only_in_full(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-for-test")

    class BrokenModel:
        def invoke(self, _messages):
            try:
                raise OSError("DNS lookup failed")
            except OSError as root:
                raise ConnectionError("api_key=sk-secret-for-test; provider connection failed") from root

    gateway = ModelGateway([
        ModelProvider("primary", BrokenModel()),
        ModelProvider("backup", QueueModel({"intent": "points"}, {"intent": "points"})),
    ])
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    token = trace_logging.current_trace.set(
        trace_logging.TraceContext("t-1", "U1001", "s-1", "test")
    )
    try:
        gateway.call([HumanMessage(content="查询积分")], lambda value: value, stage="navigation")
    finally:
        trace_logging.current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)

    failed = next(event for event in capture.events
                  if event["event"] == "model.attempt.finished" and event["status"] == "failed")
    assert failed["error_message"] == "api_key=[REDACTED]; provider connection failed"
    assert failed["cause_chain"][-1] == {"class": "OSError", "message": "DNS lookup failed"}
    assert "sk-secret-for-test" not in json.dumps(capture.events)

    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "summary")
    capture.events.clear()
    token = trace_logging.current_trace.set(
        trace_logging.TraceContext("t-2", "U1001", "s-1", "test")
    )
    trace_logging._logger.addHandler(capture)
    try:
        gateway.call([HumanMessage(content="查询积分")], lambda value: value, stage="navigation")
    finally:
        trace_logging.current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)
    failed = next(event for event in capture.events
                  if event["event"] == "model.attempt.finished" and event["status"] == "failed")
    assert "error_message" not in failed and "cause_chain" not in failed


def test_gateway_wait_timeout_reports_budget_instead_of_empty_exception(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")

    class SlowModel:
        def invoke(self, _messages):
            time.sleep(0.1)
            return AIMessage(content="late")

    gateway = ModelGateway([
        ModelProvider("primary", SlowModel(), timeout=0.01),
        ModelProvider("backup", QueueModel({"intent": "points"})),
    ], total_timeout=0.1)
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    token = trace_logging.current_trace.set(
        trace_logging.TraceContext("t-3", "U1001", "s-1", "test")
    )
    try:
        gateway.call([HumanMessage(content="查询积分")], lambda value: value, stage="navigation")
        deadline = time.monotonic() + 1
        while (not any(event["event"] == "model.attempt.late_result" for event in capture.events)
               and time.monotonic() < deadline):
            time.sleep(0.01)
    finally:
        trace_logging.current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)
    failed = next(event for event in capture.events
                  if event["event"] == "model.attempt.finished" and event["status"] == "failed")
    assert failed["error_type"] == "timeout"
    assert failed["timeout_source"] == "gateway_wait"
    assert "0.01 秒" in failed["error_message"]
    late = next(event for event in capture.events if event["event"] == "model.attempt.late_result")
    assert late["trace_id"] == "t-3" and late["status"] == "success_ignored"


def test_late_provider_exception_keeps_original_trace_and_diagnostic(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "full")

    class SlowFailure:
        def invoke(self, _messages):
            time.sleep(0.04)
            raise ConnectionError("DNS resolution failed")

    gateway = ModelGateway([ModelProvider("primary", SlowFailure(), timeout=0.01)],
                           total_timeout=0.1)
    capture = EventCapture()
    trace_logging._logger.addHandler(capture)
    token = trace_logging.current_trace.set(
        trace_logging.TraceContext("late-1", "U1001", "s-1", "test")
    )
    try:
        with pytest.raises(GatewayFailure):
            gateway.call([HumanMessage(content="查询积分")], lambda value: value, stage="navigation")
        deadline = time.monotonic() + 1
        while (not any(event["event"] == "model.attempt.late_result" for event in capture.events)
               and time.monotonic() < deadline):
            time.sleep(0.01)
    finally:
        trace_logging.current_trace.reset(token)
        trace_logging._logger.removeHandler(capture)
    late = next(event for event in capture.events if event["event"] == "model.attempt.late_result")
    assert late["trace_id"] == "late-1"
    assert late["error_type"] == "connection"
    assert late["error_message"] == "DNS resolution failed"


def test_optional_trace_file_uses_private_permissions(tmp_path):
    target = tmp_path / "logs" / "trace.jsonl"
    env = {**os.environ, "BLOOD_AGENT_TRACE_MODE": "full", "BLOOD_AGENT_TRACE_FILE": str(target)}
    script = (
        "from blood_agents.trace_logging import TraceContext, current_trace, trace_event\n"
        "token = current_trace.set(TraceContext('turn-1', 'U1001', 'S001', 'test'))\n"
        "trace_event('probe', full={'user_question': '我要预约'})\n"
        "current_trace.reset(token)\n"
    )
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                   env=env, check=True, capture_output=True, text=True)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    record = json.loads(target.read_text(encoding="utf-8").strip())
    assert record["trace_id"] == "turn-1" and record["user_question"] == "我要预约"


def test_trace_viewer_filters_and_expands_raw_values(tmp_path, capsys):
    target = tmp_path / "trace.jsonl"
    records = [
        {"ts": "t0", "event": "turn.start", "event_zh": "收到用户问题",
         "trace_id": "turn-1", "session_id": "same-session", "user_id": "U1001"},
        {"ts": "t1", "event": "model.request", "event_zh": "发送大模型输入",
         "trace_id": "turn-1", "session_id": "same-session", "user_id": "U1001",
         "stage": "navigation", "stage_zh": "主导航",
         "model_input": [{"role": "human", "content": "第一轮问题"}]},
        {"ts": "t1.5", "event": "turn.start", "event_zh": "收到用户问题",
         "trace_id": "turn-2", "session_id": "same-session", "user_id": "U1001"},
        {"ts": "t2", "event": "model.response", "event_zh": "收到大模型原始输出",
         "trace_id": "turn-2", "session_id": "same-session", "user_id": "U1001",
         "model_output": "第二轮原始输出"},
        {"ts": "t3", "event": "model.response", "event_zh": "收到大模型原始输出",
         "trace_id": "turn-1", "session_id": "same-session", "user_id": "U1001",
         "model_output": "第一轮超时后才返回"},
    ]
    target.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records),
                      encoding="utf-8")
    assert trace_viewer_main(["--file", str(target)]) == 0
    output = capsys.readouterr().out
    assert "收到大模型原始输出" in output and "第一轮问题" not in output
    assert "turn-1" not in output
    assert "第二轮原始输出" not in output
    assert trace_viewer_main(["--file", str(target), "--session-id", "same-session",
                              "--raw"]) == 0
    output = capsys.readouterr().out
    assert "第一轮问题" in output and "第二轮原始输出" in output
