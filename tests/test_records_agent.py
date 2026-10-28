"""献血记录子图：只允许旧接口本轮返回的登录用户记录进入模型回复。"""

import json
import sqlite3

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents import api as api_module
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from blood_agents.web import ChatApplication
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


class RecordsNavigation:
    def invoke(self, _messages):
        return AIMessage(content='{"intent":"records"}')


class RecordsReply:
    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        result = facts["result"]
        if result["status"] == "records_query_failed":
            answer = "献血记录查询暂不可用，请稍后重试；目前无法确认您的献血记录。"
        elif result["status"] == "no_records":
            answer = "这次查询没有找到您的献血记录。"
        else:
            summary = result["summary"]
            first = result["records"][0]
            answer = (f"查到 {summary['donation_count']} 条献血记录。最近一条是 {first['donated_at']}"
                      f" 在{first['site']}捐献 {first['donation_amount']} {first['donation_unit']}。")
            if summary["total_volume_ml"]:
                answer += f"按毫升统计的合计为 {summary['total_volume_ml']} 毫升。"
        return AIMessage(content=answer)


def setup(api_factory=MockLegacyBloodAPI, reply_model=None):
    repo = BloodRepository()
    repo.seed_demo_data()
    api = api_factory(repo)
    graph = ConversationGraph(build_graph(
        repo, IntentRouter(llm=RecordsNavigation(), response_llm=reply_model or RecordsReply(),
                           risk_llm=False),
        legacy_api=api, checkpointer=MemorySaver(),
    ))
    return repo, api, graph


def ask(graph, text="查询我的献血记录", *, user="U1001", session="records-test"):
    return graph.invoke(
        {"user_id": user, "messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": session}},
    )


def test_records_are_queried_through_mock_interface_and_model_summarizes_receipt():
    _, api, graph = setup()
    state = ask(graph)
    assert state["intent"] == "records"
    assert state["result"]["status"] == "records_found"
    assert state["result"]["summary"] == {
        "donation_count": 2, "total_volume_ml": 700, "whole_blood_count": 2,
        "component_donation_count": 0, "component_total_units": 0, "earned_points": 700,
    }
    assert state["result"]["records"] == api.get_donation_records("U1001")["data"]["records"]
    assert state["result"]["records"][0]["donation_type"] == "全血"
    assert state["result"]["records"][0]["collection_org"] == "演示市血液中心"
    assert state["result"]["records"][0]["blood_type"] == "A"
    assert state["model_info"]["navigation"]["mode"] == "llm"
    assert state["response_info"]["mode"] == "llm"
    assert "2025-12-15" in state["messages"][-1].content
    assert [item["tool"] for item in state["tool_trace"]] == ["query_records"]
    assert all(not item["mutating"] for item in state["tool_trace"])


def test_component_record_keeps_treatment_units_separate_from_milliliters():
    _, api, graph = setup()
    state = ask(graph, user="U1003", session="component-record")
    record = state["result"]["records"][0]
    assert record == api.get_donation_records("U1003")["data"]["records"][0]
    assert record["donation_type"] == "单采血小板"
    assert record["donation_amount"] == 2
    assert record["donation_unit"] == "治疗量"
    assert record["volume_ml"] is None
    assert state["result"]["summary"]["total_volume_ml"] == 0
    assert state["result"]["summary"]["component_total_units"] == 2
    assert "2 治疗量" in state["messages"][-1].content


def test_page_seeded_donated_appointment_has_one_matching_mock_record():
    repo = BloodRepository()
    repo.seed_demo_data(include_appointment_samples=True)
    repo.seed_demo_data(include_appointment_samples=True)
    donated = next(item for item in repo.list_appointments("U1002") if item.status == "donated")
    records = MockLegacyBloodAPI(repo).get_donation_records("U1002")["data"]["records"]
    assert len(records) == 1
    assert records[0]["donated_at"] == donated.appointment_date
    assert records[0]["site"] == donated.site
    assert records[0]["donation_amount"] == donated.donation_amount
    assert len(repo.list_records("U1001")) == 2


def test_existing_record_database_is_migrated_without_inventing_missing_metadata(tmp_path):
    db_path = str(tmp_path / "old-records.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE donation_records (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id TEXT NOT NULL, donated_at TEXT NOT NULL, site TEXT NOT NULL,"
            "volume_ml INTEGER NOT NULL, points INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO donation_records(user_id,donated_at,site,volume_ml,points)"
            " VALUES ('U1002','2020-05-01','原有献血点',200,100)"
        )
    repo = BloodRepository(db_path)
    repo.seed_demo_data()
    old = repo.list_records("U1002")[0]
    assert old.site == "原有献血点"
    assert old.donation_amount == 200 and old.donation_unit == "mL"
    assert old.collection_org is None and old.donation_type is None
    assert old.blood_type is None and old.rhd_type is None
    repo.seed_demo_data()
    assert len(repo.list_records("U1003")) == 1


def test_records_agent_does_not_expose_unlisted_personal_fields_from_mock_receipt():
    class ExtraFieldsAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            receipt = super().get_donation_records(user_id)
            receipt["data"]["records"][0].update({
                "name": "合成用户", "mobile": "13800005678", "id_card": "310101199001011234",
            })
            return receipt

    _, _, graph = setup(ExtraFieldsAPI)
    state = ask(graph, session="record-field-whitelist")
    assert all("mobile" not in item and "id_card" not in item and "name" not in item
               for item in state["result"]["records"])


def test_empty_records_do_not_borrow_other_users_history_or_assume_failed_query():
    _, _, graph = setup()
    state = ask(graph, "查询 U1001 的献血记录", user="U1002", session="other-user")
    assert state["result"]["status"] == "no_records"
    assert state["result"]["success"] is True
    assert state["result"]["records"] == []
    assert state["result"]["summary"]["donation_count"] == 0
    assert "2025-12-15" not in state["messages"][-1].content
    assert state["response_info"]["mode"] == "llm"
    assert ask(graph, user="UNKNOWN", session="unknown-user")["result"]["status"] == "records_query_failed"


def test_failed_or_malformed_legacy_receipt_cannot_be_presented_as_empty_records():
    class FailedAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            del user_id
            return {"success": False, "code": "UPSTREAM_DOWN", "message": "上游暂不可用"}

    class MalformedAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            del user_id
            return {"success": True, "code": "OK", "data": {"records": None}}

    class MalformedDataAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            del user_id
            return {"success": True, "code": "OK", "data": ["非预期的第三方结构"]}

    class RaisingAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            del user_id
            raise TimeoutError("旧接口超时")

    for api_factory, expected_code in (
        (FailedAPI, "UPSTREAM_DOWN"), (MalformedAPI, "INVALID_RESPONSE"),
        (MalformedDataAPI, "INVALID_RESPONSE"),
        (RaisingAPI, "UPSTREAM_ERROR"),
    ):
        _, _, graph = setup(api_factory)
        state = ask(graph, session=expected_code)
        assert state["result"]["status"] == "records_query_failed"
        assert state["result"]["code"] == expected_code
        assert "records" not in state["result"]
        assert state["response_info"]["mode"] == "llm"
        assert "暂不可用" in state["messages"][-1].content


def test_model_cannot_invent_history_when_result_is_empty_or_failed():
    class InventedReply:
        def invoke(self, _messages):
            return AIMessage(content="查到您已献血 2 次，2025-12-15 在市中心献血屋献血 400 毫升。")

    class FailedAPI(MockLegacyBloodAPI):
        def get_donation_records(self, user_id):
            del user_id
            return {"success": False, "code": "UPSTREAM_DOWN", "message": "上游暂不可用"}

    for api_factory, user in ((MockLegacyBloodAPI, "U1002"), (FailedAPI, "U1001")):
        _, _, graph = setup(api_factory, InventedReply())
        state = ask(graph, user=user, session="invented-" + user)
        assert state["response_info"]["mode"] == "unavailable"
        assert "2025-12-15" not in state["messages"][-1].content


def test_empty_result_allows_natural_negative_wording_without_false_block():
    class NegativeReply:
        def invoke(self, _messages):
            return AIMessage(content=(
                "未查询到您曾献血的记录；这仅表示当前账号本轮结果为空。"
                "如果您确认自己有过献血经历，可以联系血站工作人员核对。"
            ))

    _, _, graph = setup(reply_model=NegativeReply())
    state = ask(graph, user="U1002", session="negative-records")
    assert state["result"]["status"] == "no_records"
    assert state["response_info"]["mode"] == "llm"
    assert "未查询到您曾献血的记录" in state["messages"][-1].content
    assert "如果您确认自己有过献血经历" in state["messages"][-1].content


def test_empty_result_retries_unsupported_cause_and_keeps_model_reply():
    class RevisingReply:
        def __init__(self):
            self.requests = []

        def invoke(self, messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                return AIMessage(content="本轮未查到记录，可能是登记信息不一致，换用其他账号登录再试。")
            assert "不要推测账号" in json.loads(messages[-1].content)["revision_feedback"]
            return AIMessage(content="当前账号本轮未查到献血记录；这不等于您从未献血。")

    model = RevisingReply()
    _, _, graph = setup(reply_model=model)
    state = ask(graph, user="U1002", session="records-cause-retry")
    assert len(model.requests) == 2
    assert state["response_info"]["mode"] == "llm"
    assert "不等于您从未献血" in state["messages"][-1].content


def test_empty_result_does_not_publish_repeated_unsupported_cause():
    class SpeculativeReply:
        def invoke(self, _messages):
            return AIMessage(content="本轮未查到记录，可能是账号与登记信息不一致。")

    _, _, graph = setup(reply_model=SpeculativeReply())
    state = ask(graph, user="U1002", session="records-cause-rejected")
    assert state["response_info"]["mode"] == "unavailable"
    assert "登记信息" not in state["messages"][-1].content


def test_model_cannot_add_a_date_outside_the_successful_receipt():
    class InventedDateReply:
        def invoke(self, _messages):
            return AIMessage(content="您在 2024-01-01 献血过。")

    _, _, graph = setup(reply_model=InventedDateReply())
    state = ask(graph, session="invented-date")
    assert state["result"]["status"] == "records_found"
    assert state["response_info"]["mode"] == "unavailable"
    assert "2024-01-01" not in state["messages"][-1].content


def test_model_cannot_add_a_volume_outside_the_successful_receipt():
    class InventedVolumeReply:
        def invoke(self, _messages):
            return AIMessage(content="累计献血 999 毫升，最近一次是 2025年12月15日。")

    _, _, graph = setup(reply_model=InventedVolumeReply())
    state = ask(graph, session="invented-volume")
    assert state["result"]["status"] == "records_found"
    assert state["response_info"]["mode"] == "unavailable"
    assert "999" not in state["messages"][-1].content


def test_model_cannot_change_component_units_or_blood_type():
    class InventedComponentFacts:
        def __init__(self, answer):
            self.answer = answer

        def invoke(self, _messages):
            return AIMessage(content=self.answer)

    for index, answer in enumerate((
        "您于 2026-01-20 捐献了 3 个治疗量的单采血小板。",
        "您于 2026-01-20 捐献了 2 个治疗量的单采血小板，血型 O型。",
        "您的献血编号是 MOCK-D999999，捐献了 2 个治疗量。",
    )):
        _, _, graph = setup(reply_model=InventedComponentFacts(answer))
        state = ask(graph, user="U1003", session=f"component-invention-{index}")
        assert state["result"]["status"] == "records_found"
        assert state["response_info"]["mode"] == "unavailable"
        assert answer not in state["messages"][-1].content


def test_api_and_gradio_callbacks_show_only_the_authenticated_users_records(monkeypatch, tmp_path):
    _, _, graph = setup()
    monkeypatch.setattr(api_module, "graph", graph)
    client = TestClient(api_module.app)
    response = client.post("/chat", json={
        "user_id": "U1001", "session_id": "records-api", "message": "查询献血记录",
    }, headers={"X-User-ID": "U1002"})
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "no_records"
    assert response.json()["response_info"]["mode"] == "llm"
    assert [item["tool"] for item in response.json()["tool_trace"]] == ["query_records"]

    chat = ChatApplication(str(tmp_path / "records-ui.db"))
    chat.graph = graph
    _, history, _, details = chat.respond("查询我的献血记录", [], "U1001", "records-ui")
    assert details["result"]["status"] == "records_found"
    assert details["response_info"]["mode"] == "llm"
    assert "700 毫升" in history[-1][1]
