"""纯 Agent 包在没有本机业务插件时仍可启动并明确处理外部接口故障。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_isolated_agent_package_starts_without_local_business(tmp_path):
    """只复制将要发布的包，在新进程验证 API/UI/图及查询失败边界。"""
    source = Path(__file__).resolve().parents[1] / "blood_agents"
    shutil.copytree(source, tmp_path / "blood_agents",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script = """
import importlib.util
from pathlib import Path

import blood_agents
from langchain_core.messages import HumanMessage

assert Path(blood_agents.__file__).resolve().is_relative_to(Path.cwd())
assert importlib.util.find_spec('local_business') is None

from blood_agents.api import app
from blood_agents.web import ChatApplication
from blood_agents.orchestration.graph import build_conversational_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.storage.repository import AgentRepository

assert app is not None
ui = ChatApplication(str(Path.cwd() / 'ui.db'))
ui.graph.close()
repo = AgentRepository(str(Path.cwd() / 'agent.db'))
graph = build_conversational_graph(
    repo, IntentRouter(response_llm=False, risk_llm=False, evidence_llm=False)
)
try:
    state = graph.invoke(
        {'user_id': 'U1001', 'messages': [HumanMessage(content='查询我的积分')]},
        {'configurable': {'thread_id': 'isolated'}, 'metadata': {'entrypoint': 'test'}},
    )
    assert state['result']['status'] == 'points_query_failed', state['result']
    assert state['result']['success'] is False
    assert 'points' not in state['result']
finally:
    graph.close()
"""
    # 子进程只继承运行必需的环境，避免本机 .env/API Key 对本测试产生依赖。
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "LANG", "LC_ALL", "PYTHONPYCACHEPREFIX"}}
    env.update({
        "BLOOD_AGENT_DB": str(tmp_path / "api.db"),
        "BLOOD_AGENT_BUSINESS_TRANSPORT": "http",
        "BLOOD_AGENT_LEGACY_API_URL": "http://127.0.0.1:1",
        "BLOOD_AGENT_LEGACY_API_TIMEOUT": "0.2",
        "BLOOD_AGENT_TRACE_MODE": "off",
    })
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
