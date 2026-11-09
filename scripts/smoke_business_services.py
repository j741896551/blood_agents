"""本机双进程烟测：Agent 图经 HTTP/MCP 调用独立业务替身。

运行：.venv312/bin/python scripts/smoke_business_services.py
只使用临时合成数据库和本地测试回复模型，不调用平台大模型或真实业务接口。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.storage.repository import AgentRepository


class PreviewReplyModel:
    """仅为本机测试交付预约预览，不生成业务成功回执。"""

    def invoke(self, messages):
        """使用本轮结构化事实生成足以测试显式确认的预览。"""
        facts = json.loads(messages[-1].content)
        if facts.get("result_status") != "awaiting_confirmation":
            raise RuntimeError("smoke model only delivers appointment previews")
        return AIMessage(content=(
            "预约尚未提交，请核对本次献血人、网点与时间："
            + "、".join(facts["required_facts"]) + "。确认无误请回复确认预约。"
        ))


def _free_loopback_port() -> int:
    """从系统分配临时本机端口，避免固定端口与现有服务冲突。"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_health(url: str, process: subprocess.Popen) -> None:
    """等待替身健康检查，不依赖固定睡眠时长。"""
    deadline = time.monotonic() + 15
    last_error = "no response"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"business service exited: {process.returncode}")
        try:
            response = httpx.get(f"{url}/health", timeout=0.5, trust_env=False)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.1)
    raise TimeoutError(f"local business service did not become healthy ({last_error})")


def _ask(graph, user_id: str, session_id: str, message: str) -> dict:
    """经真实 LangGraph 对话入口执行一轮，保留用户/会话隔离。"""
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": session_id}, "metadata": {"entrypoint": "smoke"}},
    )


def main() -> None:
    """启动独立业务进程并核对 HTTP 查询、确认写入与 MCP 只读查询。"""
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="blood-agent-services-") as temporary:
        directory = Path(temporary)
        port = _free_loopback_port()
        url = f"http://127.0.0.1:{port}"
        # 本机替身不需要平台模型密钥，也不应从 Agent 进程继承它们。
        service_env = {key: value for key, value in os.environ.items()
                       if key in {"PATH", "HOME", "LOGNAME", "SHELL", "TERM", "USER",
                                  "PYTHONPYCACHEPREFIX"}}
        service_env.update({
            "BLOOD_AGENT_LOCAL_API_HOST": "127.0.0.1",
            "BLOOD_AGENT_LOCAL_API_PORT": str(port),
            "BLOOD_AGENT_LOCAL_BUSINESS_DB": str(directory / "business.db"),
            "PYTHONPATH": str(project_root),
        })
        service_log = (directory / "business-service.log").open("w+", encoding="utf-8")
        service = subprocess.Popen(
            [sys.executable, "-m", "local_business.http_service"],
            cwd=project_root, env=service_env,
            stdout=service_log, stderr=subprocess.STDOUT,
        )
        try:
            try:
                _wait_health(url, service)
            except (RuntimeError, TimeoutError) as exc:
                service_log.flush()
                service_log.seek(0)
                raise RuntimeError(f"{exc}; service log: {service_log.read()[-1500:]}") from exc
            os.environ["NO_PROXY"] = "127.0.0.1,localhost"
            os.environ.update({
                "BLOOD_AGENT_BUSINESS_TRANSPORT": "http",
                "BLOOD_AGENT_LEGACY_API_URL": url,
                "BLOOD_AGENT_DB": str(directory / "agent.db"),
                "BLOOD_AGENT_TRACE_MODE": "off",
            })
            repo = AgentRepository(str(directory / "agent.db"))
            graph = build_conversational_graph(repo, IntentRouter(
                response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False,
            ))
            try:
                points = _ask(graph, "U1001", "points", "查询我的积分")
                assert points["result"]["status"] == "points_found", points["result"]
                assert points["result"]["success"] is True
                site_data = httpx.get(f"{url}/v1/users/U1001/sites-and-slots", timeout=3).json()
                first_site = site_data[0]
                first_slot = first_site["slots"][0]
                date = first_slot["date"]
                time_slot = first_slot["time_slot"]
                site = first_site["site"]
                preview = _ask(graph, "U1001", "appointment", f"预约 {date} {time_slot}到{site}")
                assert preview["result"]["status"] == "awaiting_confirmation", preview["result"]
                submitted = _ask(graph, "U1001", "appointment", "确认预约")
                assert submitted["result"]["success"] is True, submitted["result"]
                appointments = httpx.get(f"{url}/v1/users/U1001/appointments", timeout=3).json()
                other_user = httpx.get(f"{url}/v1/users/U1002/appointments", timeout=3).json()
                appointment_id = submitted["result"]["data"]["appointment_id"]
                assert any(row["appointment_id"] == appointment_id
                           for row in appointments["data"]["appointments"])
                assert all(row["appointment_id"] != appointment_id
                           for row in other_user["data"]["appointments"])
            finally:
                graph.close()

            os.environ["BLOOD_AGENT_BUSINESS_TRANSPORT"] = "mcp"
            direct_records = httpx.get(f"{url}/v1/users/U1002/donation-records",
                                       timeout=3, trust_env=False)
            assert direct_records.status_code == 200
            # MCP SDK 只继承少量安全环境变量；先用同样的最小环境探测子进程 HTTP。
            child_env = {key: value for key, value in os.environ.items()
                         if key in {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}}
            child_env.update({
                "BLOOD_AGENT_LEGACY_API_URL": url,
                "BLOOD_AGENT_LEGACY_API_TIMEOUT": "3",
            })
            child_code = (
                "from blood_agents.business.legacy_http import LegacyHTTPClient; "
                "import os; c=LegacyHTTPClient("
                "os.environ['BLOOD_AGENT_LEGACY_API_URL'],timeout=3); "
                "assert c.get_donation_records('U1002')['success']; c.close()"
            )
            child_probe = subprocess.run(
                [sys.executable, "-c", child_code],
                cwd=project_root, env=child_env, capture_output=True, text=True,
                timeout=8, check=False,
            )
            assert child_probe.returncode == 0, child_probe.stderr[-1000:]
            mcp_graph = build_conversational_graph(AgentRepository(str(directory / "agent.db")),
                                                   IntentRouter())
            try:
                records = _ask(mcp_graph, "U1002", "records", "查询我的献血记录")
                assert records["result"]["status"] == "records_found", records["result"]
                assert records["result"]["success"] is True
                assert any(item["backend"] == "mcp_stdio" for item in records["tool_trace"])
            finally:
                mcp_graph.close()
            print("PASS: HTTP 查询、确认后预约写入、跨用户隔离、MCP stdio 查询")
        except Exception:
            service_log.flush()
            service_log.seek(0)
            print("Business service log tail:\n" + service_log.read()[-2500:], file=sys.stderr)
            raise
        finally:
            service.terminate()
            try:
                service.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(timeout=5)
            service_log.close()


if __name__ == "__main__":
    main()
