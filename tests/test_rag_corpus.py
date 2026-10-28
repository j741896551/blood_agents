"""多来源演示知识库的出处、分块与检索边界。"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.evals.evaluation import evaluate_retrieval
from blood_agents.knowledge.rag import default_retriever, load_approved_corpus
from blood_agents.llm.reply_composer import _validate_answer
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


def test_default_corpus_contains_only_public_sources():
    chunks = load_approved_corpus()
    assert len({doc.source for doc in chunks}) == 10
    assert len({doc.source for doc in chunks if doc.metadata["source_kind"] == "public"}) == 10
    assert not any(doc.metadata["source_kind"] == "synthetic" for doc in chunks)
    assert all(doc.metadata["title"] and doc.metadata["checked_at"]
               and doc.metadata["provenance"] for doc in chunks)
    hits = default_retriever().search("献血前可以空腹吗")
    assert hits
    assert any("hbhr.gov.cn" in hit["source"] and hit["title"] for hit in hits)
    interval_hits = default_retriever().search("我下次什么时候可以献血")
    assert any("不同献血类型的最短间隔" in hit["title"]
               and "180 天" in hit["text"] for hit in interval_hits)


@pytest.mark.parametrize(("question", "source_suffix"), [
    ("献血前要做哪些健康征询和检查", "ea9d8d2493f84cbf802f6ebb6155fa55.shtml"),
    ("全血和单采血小板有什么不同", "a731e5672ddc45cea7c85ecf90821889.shtml"),
    ("献血后24小时能剧烈运动吗", "2026070215022620516/index.shtml"),
    ("全国电子无偿献血证在哪里查询", "df0e888f92e642c68c13cb2976503b97.shtml"),
    ("用血费用减免能线上申请吗", "43f8d112f86f428982346ea219801f0b.shtml"),
    ("全国无偿献血表彰奖励是什么", "70b1323952aa42f9bdbd5237af3a1595.shtml"),
    ("献出的血液会经过哪些检测和成分制备环节", "ea9d8d2493f84cbf802f6ebb6155fa55.shtml"),
    ("献血前要知情同意和健康征询吗", "19aefe515bbc4c6d86bb64a389195a38.shtml"),
])
def test_new_official_topics_are_searchable(question, source_suffix):
    assert any(hit["source"].endswith(source_suffix)
               for hit in default_retriever().search(question, top_k=6))


def test_corpus_rejects_missing_source_and_duplicate_ids(tmp_path):
    source = {
        "id": "one", "title": "演示", "source": "https://example.org/article",
        "published_at": "2025-01-01", "checked_at": "2026-10-15",
        "provenance": "人工摘要", "text": "献血前准备",
    }
    file = tmp_path / "corpus.json"
    file.write_text(json.dumps([{**source, "source": "not-a-url"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="HTTPS"):
        load_approved_corpus(file)
    file.write_text(json.dumps([source, source]), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_approved_corpus(file)


def test_corpus_rejects_path_escape_and_wrong_synthetic_source(tmp_path):
    record = {
        "id": "one", "title": "模拟流程", "source": "demo://blood-station/test",
        "source_kind": "synthetic", "published_at": "2026-10-16", "checked_at": "2026-10-16",
        "provenance": "虚构演示", "path": "../secret.md",
    }
    file = tmp_path / "corpus.json"
    file.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="越出"):
        load_approved_corpus(file)
    record["path"] = "document.md"
    record["source"] = "https://example.org/fake"
    file.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="demo"):
        load_approved_corpus(file)


def test_multisource_retrieval_evaluation_uses_distinct_expected_sources():
    cases = json.loads((Path(__file__).parents[1] / "evals/dataset.json")
                       .read_text(encoding="utf-8"))["retrieval"]
    assert len({case["source"] for case in cases if case["source"]}) == 7
    report = evaluate_retrieval(cases, default_retriever(), top_k=4)
    assert report["count"] == 10
    assert report["positive_count"] == 8
    assert report["negative_count"] == 2
    assert report["recall@4"] == 1.0
    assert report["no_evidence_rate"] == 1.0


def test_unrelated_question_does_not_promote_weak_character_overlap_to_evidence():
    retriever = default_retriever()
    assert retriever.search("今天上海天气如何") == []
    assert retriever.search("北京地铁末班车几点") == []


def test_synthetic_source_reply_must_disclose_demo_status():
    hit = {"source": "demo://blood-station/points-delivery", "source_kind": "synthetic",
           "title": "合成流程", "text": "下单前确认收货信息。"}
    answer = f"本血站要求确认收货信息。来源：{hit['source']}"
    with pytest.raises(TypeError, match="演示性质"):
        _validate_answer(answer, (hit["source"],), "faq",
                         {"citations": [hit["source"]]}, retrieved_context=[hit])
    disclosed = answer.replace("本血站要求", "模拟系统演示流程要求")
    assert _validate_answer(disclosed, (hit["source"],), "faq",
                            {"citations": [hit["source"]]}, retrieved_context=[hit]) == disclosed


def test_faq_reply_may_cite_one_selected_source_but_not_invent_another():
    hits = [
        {"doc_id": "a", "source": "https://example.org/a", "title": "献血注意事项",
         "source_kind": "public", "text": "请携带身份证件。"},
        {"doc_id": "b", "source": "https://example.org/b", "title": "献血流程",
         "source_kind": "public", "text": "到站先登记。"},
    ]
    result = {"citations": [hit["source"] for hit in hits]}
    answer = "请携带身份证件。来源：https://example.org/a"
    assert _validate_answer(answer, (), "faq", result, retrieved_context=hits) == answer
    with pytest.raises(TypeError, match="未注明"):
        _validate_answer("请携带身份证件。", (), "faq", result, retrieved_context=hits)
    with pytest.raises(TypeError, match="之外"):
        _validate_answer(answer + "、https://example.org/invented", (), "faq", result,
                         retrieved_context=hits)
    with pytest.raises(TypeError, match="不一致"):
        _validate_answer(answer, (), "faq",
                         {"citations": [hits[0]["source"], "https://example.org/invented"]},
                         retrieved_context=hits)


def test_faq_reply_accepts_known_title_but_synthetic_title_needs_disclosure():
    hit = {"doc_id": "demo", "source": "demo://blood-station/points-delivery",
           "title": "积分兑换流程", "source_kind": "synthetic",
           "text": "下单前展示礼品和收货信息。"}
    result = {"citations": [hit["source"]]}
    answer = "演示系统下单前会展示礼品和收货信息。来源：《积分兑换流程》"
    assert _validate_answer(answer, (), "faq", result, retrieved_context=[hit]) == answer
    with pytest.raises(TypeError, match="演示性质"):
        _validate_answer("下单前会展示礼品和收货信息。来源：《积分兑换流程》", (),
                         "faq", result, retrieved_context=[hit])


def test_default_faq_path_passes_source_and_score_to_reply_model():
    class Navigation:
        def invoke(self, _messages):
            return AIMessage(content='{"intent":"faq"}')

    class Reply:
        def __init__(self):
            self.facts = None

        def invoke(self, messages):
            self.facts = json.loads(messages[-1].content)
            sources = list(dict.fromkeys(hit["source"] for hit in self.facts["retrieved_context"]))
            return AIMessage(content=f"检索资料提示不要空腹献血。来源：{'、'.join(sources)}")

    reply = Reply()
    repo = BloodRepository()
    graph = build_graph(
        repo, IntentRouter(llm=Navigation(), response_llm=reply, risk_llm=False,
                           evidence_llm=False),
        checkpointer=MemorySaver(),
    )
    state = graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content="献血前可以空腹吗？")]},
        {"configurable": {"thread_id": "multi-source-faq"}},
    )
    assert state["result"]["status"] == "grounded"
    assert state["response_info"]["mode"] == "llm"
    hit = reply.facts["retrieved_context"][0]
    assert hit["source"] in state["result"]["citations"]
    assert hit["title"] == "献血前后的注意事项摘要"
    assert hit["score"] > 0
