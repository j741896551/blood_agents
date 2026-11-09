"""真实平台探测器的离线回归；绝不发送外部 API 请求。"""

import json
from threading import Barrier
from types import SimpleNamespace

from blood_agents.evals import provider_probe


class Reply:
    def __init__(self):
        self.content = "OK"
        self.usage_metadata = {"input_tokens": 17, "output_tokens": 2}


def test_parallel_probe_is_bounded_and_records_latency_without_reply_text():
    arrived = Barrier(4)

    class Model:
        def invoke(self, messages):
            assert "合成的 API 连通性测试" in messages[-1].content
            arrived.wait(timeout=5)
            return Reply()

    report = provider_probe.run_probe(Model(), provider="primary", parallel=4)
    assert report["requests"] == report["success"] == 4
    assert report["p50_ms"] is not None and report["p95_ms"] is not None
    assert all(item["input_tokens"] == 17 for item in report["results"])
    assert "OK" not in json.dumps(report)


def test_probe_reports_safe_error_class_without_exception_message():
    class Model:
        def invoke(self, messages):
            raise ConnectionError("secret token in upstream error")

    report = provider_probe.run_probe(Model(), provider="backup", parallel=1)
    assert report["success"] == 0
    assert report["results"][0]["error_type"] == "connection"
    assert "secret token" not in json.dumps(report)


def test_cli_is_dry_run_by_default_and_execute_caps_output(monkeypatch, capsys):
    calls = []

    class Model:
        model_name = "demo-model"

        def bind(self, **kwargs):
            calls.append(("bind", kwargs))
            return self

        def invoke(self, messages):
            calls.append(("invoke", messages[-1].content))
            return Reply()

    provider = SimpleNamespace(name="primary", model=Model(), timeout=3)
    monkeypatch.setattr(provider_probe.IntentRouter, "from_env",
                        lambda: SimpleNamespace(llm=SimpleNamespace(providers=[provider])))

    assert provider_probe.main(["--provider", "primary", "--parallel", "2"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["status"] == "dry_run" and dry["plan"]["requests"] == 2
    assert calls == []

    assert provider_probe.main(["--provider", "primary", "--parallel", "2", "--execute"]) == 0
    live = json.loads(capsys.readouterr().out)
    assert live["report"]["success"] == 2
    assert calls[0] == ("bind", {"max_tokens": 128})
    assert sum(kind == "invoke" for kind, _ in calls) == 2
