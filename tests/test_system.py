from datetime import datetime, timedelta
from threading import Barrier, Thread
from zoneinfo import ZoneInfo

from blood_agents.graph import build_conversational_graph, build_graph
from blood_agents.rag import default_retriever
from blood_agents.router import IntentRouter
from blood_agents.tools import build_business_tools
from langchain_core.messages import HumanMessage
from preview_reply import PreviewReplyModel

from blood_agents.domains.ocr_service import OCRService
from blood_agents.evals.evaluation import evaluate_retrieval, evaluate_routing
from local_business.api import MockLegacyBloodAPI
from local_business.repository import BloodRepository


def setup_graph():
    repo = BloodRepository()
    repo.seed_demo_data()
    return repo, build_graph(repo, IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False))


def ask(graph, text):
    return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": "U1001"})


def test_records_route_and_summary():
    _, graph = setup_graph()
    result = ask(graph, "查询我的献血记录")
    assert result["intent"] == "records"
    assert sum(item["volume_ml"] for item in result["result"]["records"]) == 700
    assert result["response_info"]["mode"] == "unavailable"


def test_appointment_requires_slots_then_succeeds():
    _, graph = setup_graph()
    incomplete = ask(graph, "我要预约献血")
    assert incomplete["result"]["status"] == "need_more_info"
    available_date = datetime.now(tz=ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    complete = ask(graph, f"预约 {available_date} 上午到市中心献血屋")
    assert complete["intent"] == "appointment"
    assert complete["result"]["status"] == "awaiting_confirmation"
    assert complete["response_info"]["mode"] == "llm"


def test_redemption_without_model_does_not_submit():
    repo, graph = setup_graph()
    first = ask(graph, "兑换 R001")
    second = ask(graph, "兑换 R001")
    assert first["result"]["status"] == "redemption_address_required"
    assert second["result"]["status"] == "redemption_address_required"
    assert first["points_state"] == second["points_state"] == {}
    assert repo.get_points("U1001") == 1200
    assert all(item["tool"] != "redeem_reward" for item in first["tool_trace"])


def test_multi_turn_appointment_collects_slots():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo, router=IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False))
    config = {"configurable": {"thread_id": "appointment-1"}}
    available_date = datetime.now(tz=ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    first = graph.invoke(
        {"messages": [HumanMessage(content=f"我要预约 {available_date}")], "user_id": "U1001"},
        config=config,
    )
    assert first["slots"]["date"] == str(available_date)
    second = graph.invoke(
        {"messages": [HumanMessage(content="上午到市中心献血屋")], "user_id": "U1001"},
        config=config,
    )
    assert second["result"]["status"] == "awaiting_confirmation"
    third = graph.invoke(
        {"messages": [HumanMessage(content="确认预约")], "user_id": "U1001"},
        config=config,
    )
    assert third["result"]["success"] is True
    assert third["response_info"]["mode"] == "unavailable"
    assert third["slots"] == {}


def test_previous_donor_is_in_preview_and_bare_yes_does_not_submit():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo, router=IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False))
    config = {"configurable": {"thread_id": "confirm-donor-self"}}

    def send(text):
        return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": "U1001"},
                            config=config)

    day = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    preview = send(f"预约 {day} 上午到市中心献血屋")
    assert preview["result"]["donor_preview"]["name"] == "小明"
    assert preview["result"]["donor_preview"]["gender"] == "男"
    acknowledged = send("是")
    assert acknowledged["result"]["status"] == "awaiting_confirmation"
    assert acknowledged["result"]["donor_confirmed"] is True
    assert repo.list_appointments("U1001") == []
    booked = send("确认预约")
    assert booked["result"]["success"] is True
    assert booked["result"]["data"]["donor_name"] == "小明"
    assert len(repo.list_appointments("U1001")) == 1


def test_other_donor_collected_previewed_submitted_and_queried():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo, router=IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False))
    config = {"configurable": {"thread_id": "confirm-donor-other"}}

    def send(text):
        return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": "U1001"},
                            config=config)

    day = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=2)
    first = send(f"预约 {day} 上午到大学城献血车")
    assert first["result"]["donor_preview"]["name"] == "小明"
    other = send("不是我献血，是帮朋友预约")
    assert other["result"]["status"] == "need_more_info"
    assert "本次献血人姓名" in other["result"]["missing"]
    assert repo.list_appointments("U1001") == []
    preview = send("献血人姓名：张三，手机号：13912345678，身份证号：310101199303031234，性别：男，A型血，全血300cc")
    assert preview["result"]["status"] == "awaiting_confirmation"
    assert preview["result"]["donor_preview"]["name"] == "张三"
    assert preview["result"]["donor_preview"]["mobile_masked"] == "139****5678"
    booked = send("确认预约")
    assert booked["result"]["success"] is True
    appointment_id = booked["result"]["data"]["appointment_id"]
    queried = send("查询我的预约信息")
    assert queried["result"]["appointments"][0]["appointment_id"] == appointment_id
    assert queried["result"]["appointments"][0]["donor_name"] == "张三"
    assert queried["result"]["appointments"][0]["donor_mobile_masked"] == "139****5678"


def test_mock_rejects_second_booking_for_same_donor_on_same_day():
    repo = BloodRepository()
    repo.seed_demo_data()
    graph = build_conversational_graph(repo, router=IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False))
    config = {"configurable": {"thread_id": "donor-duplicate"}}

    def send(text):
        return graph.invoke({"messages": [HumanMessage(content=text)], "user_id": "U1001"},
                            config=config)

    day = datetime.now(ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    send(f"预约 {day} 上午到市中心献血屋")
    assert send("确认预约")["result"]["success"] is True
    send(f"预约 {day} 下午到市中心献血屋")
    rejected = send("确认预约")
    assert rejected["result"]["success"] is False
    assert rejected["result"]["code"] == "ALREADY_BOOKED"
    assert len(repo.list_appointments("U1001")) == 1


def test_mock_booking_persists_across_repository_reopen(tmp_path):
    db_path = str(tmp_path / "blood.db")
    repo = BloodRepository(db_path)
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    slot = api.site_slots[1]["slots"][0]
    payload = {
        "user_id": "U1001", "site": "大学城献血车", "date": slot["date"],
        "time_slot": slot["time_slot"], "slot_id": slot["slot_id"],
        "donor": {"kind": "other", "name": "张三", "mobile": "13912345678",
                  "id_card": "310101199303031234", "gender": "男", "blood_type": "A"},
        "donation_type": "全血", "donation_amount": 300,
    }
    first = api.submit_appointment(payload, "persist-other-donor")
    assert first["success"] is True
    retry = api.submit_appointment(payload, "persist-other-donor")
    assert retry["duplicate"] is True
    assert retry["data"]["appointment_id"] == first["data"]["appointment_id"]
    assert api.site_slots[1]["slots"][0]["remaining"] == 2

    reopened = MockLegacyBloodAPI(BloodRepository(db_path))
    found = reopened.get_user_appointments("U1001")["data"]["appointments"]
    assert found[0]["appointment_id"] == first["data"]["appointment_id"]
    assert found[0]["donor_name"] == "张三"
    assert found[0]["donor_mobile_masked"] == "139****5678"
    assert reopened.submit_appointment(payload, "another-attempt")["code"] == "ALREADY_BOOKED"


def test_concurrent_mock_submissions_do_not_overbook_last_slot():
    repo = BloodRepository()
    repo.seed_demo_data()
    api = MockLegacyBloodAPI(repo)
    slot = api.site_slots[1]["slots"][0]
    slot["remaining"] = 1
    barrier = Barrier(3)
    results = []

    def submit(name, mobile):
        payload = {
            "user_id": "U1001", "site": "大学城献血车", "date": slot["date"],
            "time_slot": slot["time_slot"], "slot_id": slot["slot_id"],
            "donor": {"kind": "other", "name": name, "mobile": mobile,
                      "id_card": "310101199303031234", "gender": "男", "blood_type": "A"},
            "donation_type": "全血", "donation_amount": 300,
        }
        barrier.wait()
        results.append(api.submit_appointment(payload, name))

    workers = [Thread(target=submit, args=("张三", "13912345678")),
               Thread(target=submit, args=("李四", "13912345679"))]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)
    assert len(results) == 2
    assert sorted(item["code"] for item in results) == ["OK", "SLOT_FULL"]
    assert slot["remaining"] == 0
    assert len(repo.list_appointments("U1001")) == 1


def test_rag_returns_citation():
    _, graph = setup_graph()
    result = ask(graph, "献血前可以空腹吗")
    assert result["intent"] == "faq"
    assert "https://www.hbhr.gov.cn/hrqxxgk/xxgkml/ggqsydwxxgk/wsjk/ddzwsy/202508/t20250821_720743.html" in result["result"]["citations"]
    assert result["response_info"]["mode"] == "unavailable"


def test_tool_ecosystem_has_more_than_ten_tools():
    repo = BloodRepository()
    registry = build_business_tools(repo, default_retriever(), OCRService(), MockLegacyBloodAPI(repo))
    assert len(registry.names()) >= 10
    assert {"submit_appointment", "knowledge_search", "ocr_extract"} <= set(registry.names())


def test_appointment_api_failure_reason_is_returned():
    repo = BloodRepository()
    repo.seed_demo_data()
    legacy_api = MockLegacyBloodAPI(repo)
    graph = build_graph(repo=repo, router=IntentRouter(
        response_llm=PreviewReplyModel(), risk_llm=False, evidence_llm=False),
        legacy_api=legacy_api, checkpointer=None)
    available_date = datetime.now(tz=ZoneInfo("Asia/Shanghai")).date() + timedelta(days=1)
    preview = ask(graph, f"预约 {available_date} 上午到市中心献血屋")
    assert preview["result"]["status"] == "awaiting_confirmation"
    for site in legacy_api.site_slots:
        for slot in site["slots"]:
            if slot["slot_id"] == "S001-AM-1":
                slot["remaining"] = 0
    # 无 checkpointer 时直接验证模拟旧接口的失败码。
    failed = legacy_api.submit_appointment(
        {
            "user_id": "U1001",
            "site": "市中心献血屋",
            "date": str(available_date),
            "time_slot": "上午",
            "slot_id": "S001-AM-1",
            "donor": {"kind": "self", "name": "小明", "mobile_masked": "138****5678"},
        },
        "failure-test",
    )
    assert failed["success"] is False
    assert failed["code"] == "SLOT_FULL"
    assert "已约满" in failed["message"]


def test_offline_evaluation_metrics():
    routing = evaluate_routing(
        [
            {"text": "查询积分", "intent": "points"},
            {"text": "识别图片", "intent": "ocr"},
        ],
        IntentRouter(),
    )
    retrieval = evaluate_retrieval(
        [{"query": "献血之后注意什么", "source": "https://www.hbhr.gov.cn/hrqxxgk/xxgkml/ggqsydwxxgk/wsjk/ddzwsy/202508/t20250821_720743.html"}],
        default_retriever(),
    )
    assert routing["accuracy"] == 1.0
    assert retrieval["recall@4"] == 1.0
