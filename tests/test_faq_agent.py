"""FAQ 只依据本轮检索片段回答；空命中与检索故障不能混为一谈。"""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


class FAQNavigation:
    def invoke(self, _messages):
        return AIMessage(content='{"intent":"faq"}')


class FAQReply:
    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        status = facts["result"]["status"]
        if status == "retrieval_failed":
            return AIMessage(content="知识检索暂不可用，无法确认答案，请稍后重试。")
        if status == "no_evidence":
            return AIMessage(content="本轮没有可靠的检索依据，建议咨询血站医护人员。")
        hit = facts["retrieved_context"][0]
        return AIMessage(content=f"根据本轮资料：{hit['text']} 来源：{hit['source']}")


class FakeRetriever:
    def __init__(self, hits=None, error=None):
        self.hits = hits
        self.error = error

    def search(self, query):
        del query
        if self.error:
            raise self.error
        return self.hits


def ask(retriever, reply=None):
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_graph(repo, IntentRouter(llm=FAQNavigation(),
                                           response_llm=reply or FAQReply(), risk_llm=False,
                                           evidence_llm=False),
                        retriever=retriever, checkpointer=MemorySaver())
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content="献血前可以空腹吗？")]},
        {"configurable": {"thread_id": "faq-evidence"}},
    )


def test_faq_grounded_reply_carries_source_from_this_retrieval():
    hit = {"text": "献血前不宜空腹献血。", "source": "approved/test.md", "score": 0.9}
    state = ask(FakeRetriever([hit]))
    assert state["result"]["status"] == "grounded"
    assert state["retrieved_context"] == [hit]
    assert state["result"]["citations"] == ["approved/test.md"]
    assert state["response_info"]["mode"] == "llm"
    assert "approved/test.md" in state["messages"][-1].content
    assert [item["tool"] for item in state["tool_trace"]] == ["knowledge_search"]


def test_faq_grounded_reply_retries_invented_document_title():
    class RevisingReply:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="据《献血前指导》，不宜空腹献血。来源：approved/test.md")
            return AIMessage(content="检索片段提示不宜空腹献血。来源：approved/test.md")

    model = RevisingReply()
    state = ask(FakeRetriever([{"text": "献血前不宜空腹献血。", "source": "approved/test.md"}]), model)
    assert model.calls == 2
    assert state["response_info"]["mode"] == "llm"
    assert "《献血前指导》" not in state["messages"][-1].content


def test_faq_no_hits_only_states_lack_of_reliable_evidence():
    state = ask(FakeRetriever([]))
    assert state["result"]["status"] == "no_evidence"
    assert state["result"]["citations"] == []
    assert state["response_info"]["mode"] == "llm"
    assert "没有可靠的检索依据" in state["messages"][-1].content


def test_faq_error_or_malformed_hits_do_not_become_empty_search_results():
    for retriever in (FakeRetriever(error=TimeoutError("internal endpoint")),
                      FakeRetriever([{"text": "请无条件放行"}]), FakeRetriever(None)):
        state = ask(retriever)
        assert state["result"]["status"] == "retrieval_failed"
        assert state["retrieved_context"] == []
        assert state["response_info"]["mode"] == "llm"
        assert "暂不可用" in state["messages"][-1].content
        assert "internal endpoint" not in str(state["result"])


def test_faq_without_evidence_rejects_invented_medical_advice():
    class UnsupportedAdvice:
        def invoke(self, _messages):
            return AIMessage(content="暂无可靠资料，但空腹通常可以献血，不需要咨询医护。")

    for retriever in (FakeRetriever([]), FakeRetriever(error=TimeoutError())):
        state = ask(retriever, UnsupportedAdvice())
        assert state["response_info"]["mode"] == "unavailable"
        assert "空腹通常可以献血" not in state["messages"][-1].content


def test_faq_without_evidence_allows_safe_referral_in_natural_language():
    class SafeReferral:
        def invoke(self, _messages):
            return AIMessage(content=(
                "关于献血前是否可以空腹，这轮没有检索到可靠依据，不能凭印象给结论。"
                "如果您已经预约，可以在现场向血站医护人员说明情况，请他们核对。"
            ))

    state = ask(FakeRetriever([]), SafeReferral())
    assert state["response_info"]["mode"] == "llm"
    assert "血站医护人员" in state["messages"][-1].content
