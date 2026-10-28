"""目录重组回归：新路径是实现，旧路径仅是向后兼容导出。"""

from pathlib import Path

from blood_agents import agents, graph, legacy_api, rag, reply_composer, repository, router
from blood_agents.domains import (
    appointment_agent,
    faq_agent,
    ocr_agent,
    points_agent,
    records_agent,
)
from blood_agents.knowledge.rag import HybridRetriever
from blood_agents.llm.reply_composer import ReplyComposer
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.storage.repository import AgentRepository


def test_package_root_has_only_entrypoints_and_compatibility_map():
    root = Path(__file__).parents[1] / "blood_agents"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py", "api.py", "cli.py", "web.py",
    }


def test_old_import_paths_point_to_new_implementations():
    assert graph.build_graph is build_graph
    assert router.IntentRouter is IntentRouter
    assert repository.AgentRepository is AgentRepository
    assert not hasattr(legacy_api, "MockLegacyBloodAPI")
    assert reply_composer.ReplyComposer is ReplyComposer
    assert rag.HybridRetriever is HybridRetriever
    for name, implementation in (
        ("appointment_agent", appointment_agent),
        ("points_agent", points_agent),
        ("records_agent", records_agent),
        ("faq_agent", faq_agent),
        ("ocr_agent", ocr_agent),
    ):
        assert getattr(agents, name) is implementation
