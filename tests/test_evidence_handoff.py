"""同一会话跨领域切换时，每轮只使用该领域本轮查询得到的证据。"""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


class Navigation:
    def invoke(self, messages):
        question = json.loads(messages[-1].content)["message"]
        intent = "records" if "记录" in question else "faq" if "空腹" in question else "ocr"
        return AIMessage(content=json.dumps({"intent": intent}))


class Reply:
    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        intent = facts["intent"]
        if intent == "records":
            return AIMessage(content="当前账号本轮未查到献血记录；这不等于您从未献血。")
        if intent == "faq":
            return AIMessage(content="本轮没有可靠的检索依据，建议咨询血站医护人员。")
        return AIMessage(content="本轮未收到图片；请提供服务端可访问的图片路径。")


class EmptyRetriever:
    def search(self, query):
        del query
        return []


def test_records_faq_and_ocr_keep_independent_facts_in_one_session():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=Navigation(), response_llm=Reply(), risk_llm=False),
        retriever=EmptyRetriever(), checkpointer=MemorySaver(),
    ))
    config = {"configurable": {"thread_id": "evidence-switch"}}

    def send(question):
        return graph.invoke({"user_id": "U1002", "messages": [HumanMessage(content=question)]}, config)

    turns = [
        ("查询我的献血记录", "records", "no_records", "query_records"),
        ("空腹能献血吗", "faq", "no_evidence", "knowledge_search"),
        ("识别图片", "ocr", "need_image", None),
        ("再查我的献血记录", "records", "no_records", "query_records"),
    ]
    for question, intent, status, tool in turns:
        state = send(question)
        assert state["intent"] == intent
        assert state["result"]["status"] == status
        assert state["response_info"]["mode"] == "llm"
        assert state["agent_handoff"]["control"] == "main"
        assert [item["tool"] for item in state["tool_trace"]] == ([tool] if tool else [])
        assert len(state["node_trace"]) > 3
    assert "献血记录" in state["messages"][-1].content
