"""验证 Agent 注册一次后，导航契约、提示词、规则和图接线同步生成。"""

import json

import pytest
from blood_agents.agent_registry import AgentRegistry, AgentSpec, default_agent_registry
from blood_agents.graph import build_graph
from blood_agents.router import IntentRouter
from langchain_core.messages import AIMessage, HumanMessage

from local_business.repository import BloodRepository


def demo_spec():
    def make_node(_dependencies):
        def node(_state):
            return {"messages": [AIMessage(content="演示子智能体已处理。")],
                    "result": {"status": "demo_completed"}}
        return node

    return AgentSpec("demo", "处理演示请求", make_node, lambda text: "演示" in text)


def test_registered_agent_drives_prompt_schema_fallback_and_graph():
    registry = default_agent_registry()
    registry.register(demo_spec())

    class Model:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content=json.dumps({"intent": "demo"}))

    model = Model()
    router = IntentRouter(model, registry=registry)
    repo = BloodRepository()
    graph = build_graph(repo, router)
    state = graph.invoke({"user_id": "U1001", "messages": [HumanMessage(content="演示一下")]})

    assert state["intent"] == "demo"
    assert state["messages"][-1].content != "演示子智能体已处理。"
    assert state["response_info"]["mode"] == "unavailable"
    assert [item["node"] for item in state["node_trace"]] == [
        "supervisor", "demo_agent_subgraph", "demo_agent", "demo_agent_reply",
        "demo_agent_handoff", "main_agent_return",
    ]
    system = model.messages[0].content
    assert "demo=处理演示请求" in system
    assert '"demo"' in system
    assert router.registry.route_map()["demo"] == "demo_subgraph"

    offline = IntentRouter(registry=registry)
    assert offline.route("演示一下") == "demo"
    with pytest.raises(RuntimeError, match="图已编译"):
        registry.register(AgentSpec("later", "稍后功能", lambda d: d, lambda text: False))


def test_registry_rejects_duplicates_reserved_intent_and_unsupported_workflow():
    registry = AgentRegistry()
    spec = demo_spec()
    registry.register(spec)
    with pytest.raises(ValueError, match="重复"):
        registry.register(spec)
    with pytest.raises(ValueError, match="保留"):
        registry.register(AgentSpec("general", "通用", spec.node_factory, spec.rule_match))
    with pytest.raises(ValueError, match="预约流程"):
        registry.register(AgentSpec("other", "其他", spec.node_factory, spec.rule_match,
                                    workflow="appointment"))


def test_unregistered_model_intent_cannot_enter_graph():
    registry = AgentRegistry()
    registry.register(demo_spec())

    class UnknownIntentModel:
        def invoke(self, _messages):
            return AIMessage(content='{"intent":"unregistered"}')

    router = IntentRouter(UnknownIntentModel(), registry=registry)
    decision, info = router.navigate("演示一下")
    assert decision.intent == "demo"
    assert info["mode"] == "rules_fallback"
    assert info["error_type"]
