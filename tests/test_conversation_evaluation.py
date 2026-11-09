"""多轮评测器本身的测试：确保断言真实生效、报告可追溯且不误报模型通过。"""

import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from pydantic import ValidationError

from blood_agents.evals.conversation_evaluation import (
    ConversationSuite,
    _heartbeat,
    _resolve_receipt_references,
    _run_suite,
    _turn_record,
    load_suite,
    main,
    run_suite,
    save_report,
)
from blood_agents.evals.conversation_reviewer import ConversationReviewer
from blood_agents.evals.excel_io import (
    INPUT_HEADERS,
    read_input_tables,
    suite_data_from_tables,
    write_workbook,
)
from blood_agents.orchestration.router import IntentRouter

DATASET = Path(__file__).parents[1] / "evals" / "conversations" / "appointment_lifecycle.json"
EXCEL_DATASET = DATASET.with_suffix(".xlsx")
BATCH_DATASET = DATASET.with_name("batch_review_example.xlsx")
FAQ_DATASET = DATASET.with_name("faq_final_reply.xlsx")


def test_fixed_dialogue_can_reuse_only_prior_mock_receipt_ids():
    prior = {"save": {"data": {"address_id": "A12345678"}}}
    assert _resolve_receipt_references(
        "把 {{address_id:save}} 设为默认地址", prior
    ) == "把 A12345678 设为默认地址"
    with pytest.raises(ValueError, match="尚无执行结果"):
        _resolve_receipt_references("{{address_id:other_case}}", prior)
    with pytest.raises(ValueError, match="没有 redemption_id"):
        _resolve_receipt_references("{{redemption_id:save}}", prior)


def test_reviewer_receives_masked_business_receipt_not_only_status_summary():
    class CaptureModel:
        def invoke(self, messages):
            self.payload = json.loads(messages[-1].content)
            return SimpleNamespace(content=json.dumps({
                "verdict": "pass", "task_completed": True, "reason": "地址已保存", "issues": [],
            }, ensure_ascii=False))

    model = CaptureModel()
    result = ConversationReviewer(model).review(goal="确认保存地址", turns=[{
        "turn_id": "confirm", "message": "确认保存地址", "answer": "地址已保存。",
        "response_info": {"mode": "llm"}, "result": {
            "status": "address_add_completed", "success": True,
            "data": {"address_id": "A12345678", "user_id": "U1001",
                     "mobile": "13800001234", "is_default": True},
        },
    }])
    assert result["status"] == "passed"
    receipt = model.payload["turns"][0]["receipt"]
    assert receipt["data"]["address_id"] == "A12345678"
    assert receipt["data"]["mobile"] == "138****1234"
    assert "user_id" not in receipt["data"]


def test_faq_reviewer_receives_only_selected_knowledge_evidence():
    class CaptureModel:
        def invoke(self, messages):
            self.payload = json.loads(messages[-1].content)
            return SimpleNamespace(content=json.dumps({
                "verdict": "pass", "task_completed": True, "reason": "回答有依据", "issues": [],
            }, ensure_ascii=False))

    model = CaptureModel()
    state = {
        "intent": "faq", "messages": [SimpleNamespace(content="间隔不少于六个月，来源：《献血法》第九条。")],
        "result": {"type": "rag", "status": "grounded", "citations": ["https://example.org/law"]},
        "retrieved_context": [{"text": "两次采集血液的间隔期不少于六个月。",
                               "title": "献血法第九条", "source": "https://example.org/law",
                               "source_kind": "public", "score": 0.75}],
        "response_info": {"mode": "llm"},
    }
    record = _turn_record(SimpleNamespace(id="t1", message="献血间隔多久", expected=None), state, [], 1.0,
                          mode="model")
    assert record["retrieved_context"][0]["text"] == "两次采集血液的间隔期不少于六个月。"
    assert ConversationReviewer(model).review(goal="依证据回答", turns=[record])["status"] == "passed"
    evidence = model.payload["turns"][0]["retrieved_context"]
    assert evidence == [{"text": "两次采集血液的间隔期不少于六个月。",
                         "title": "献血法第九条", "source": "https://example.org/law",
                         "source_kind": "public"}]

    state["result"] = {"type": "rag", "status": "no_evidence", "citations": []}
    state["retrieved_context"] = []
    empty = _turn_record(SimpleNamespace(id="t2", message="我能献血吗", expected=None), state, [], 1.0,
                         mode="model")
    ConversationReviewer(model).review(goal="无证据拒答", turns=[empty])
    assert model.payload["turns"][0]["retrieved_context"] == []


def test_example_runs_isolated_multiturn_business_assertions(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    report = _run_suite(load_suite(DATASET), mode="offline", review="none")
    assert report["summary"]["cases_total"] == 2
    assert report["summary"]["turns_total"] == 11
    assert report["summary"]["llm_replies"] == 0
    # 无模型时预览不会送达，旧的“确认后写入”断言必须失败，不能误报闭环通过。
    assert report["summary"]["cases_passed"] == 0
    assert not report["summary"]["structural_passed"]
    assert report["summary"]["turns_passed"] < 11
    turns = report["cases"][0]["turns"]
    site = next(turn for turn in turns if turn["turn_id"] == "site")
    submit = next(turn for turn in turns if turn["turn_id"] == "submit")
    cancelled = next(turn for turn in turns if turn["turn_id"] == "cancel_confirm")
    assert site["result"]["status"] == "preview_unavailable"
    assert site["response_info"]["mode"] == "unavailable"
    assert not site["slots"].get("preview_delivered")
    assert submit["result"].get("success") is not True
    assert all(item["tool"] != "submit_appointment" for item in submit["tool_trace"])
    assert all(item["tool"] != "cancel_appointment" for item in cancelled["tool_trace"])
    checkin = next(turn for turn in report["cases"][1]["turns"]
                   if turn["turn_id"] == "checkin_request")
    assert checkin["result"]["status"] == "preview_unavailable"
    assert not checkin["slots"].get("preview_delivered")
    assert all(
        item["tool"] not in {"submit_appointment", "cancel_appointment", "checkin_appointment"}
        for case in report["cases"] for turn in case["turns"] for item in turn["tool_trace"]
    )
    assert submit["trace_id"] != cancelled["trace_id"]
    json_path, xlsx_path = save_report(report, tmp_path / "reports")
    assert not json.loads(json_path.read_text(encoding="utf-8"))["summary"]["structural_passed"]
    assert not report["summary"]["passed"]  # 离线诊断不能成为模型验收通过
    with zipfile.ZipFile(xlsx_path) as workbook:
        names = workbook.namelist()
        assert "xl/worksheets/sheet2.xml" in names
        overview = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "用例内容指纹（SHA-256）" in overview
        assert report["suite_sha256"] in overview
        dialogue = workbook.read("xl/worksheets/sheet2.xml").decode("utf-8")
        assert "确认取消" in dialogue
        assert "对话结果" in workbook.read("xl/workbook.xml").decode("utf-8")
    assert stat.S_IMODE(json_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(xlsx_path.stat().st_mode) == 0o600


def test_batch_excel_runs_five_independent_multiturn_conversations(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = load_suite(BATCH_DATASET)
    assert len(suite.cases) == 5
    assert all(case.review_goal for case in suite.cases)
    report = _run_suite(suite, mode="offline", review="none")
    assert report["summary"]["cases_total"] == 5
    assert report["summary"]["turns_total"] == 18
    assert not report["summary"]["structural_passed"]
    assert not report["summary"]["passed"]
    assert report["summary"]["review_passed"] is None
    assert len({case["session_id"] for case in report["cases"]}) == 5


def test_faq_excel_suite_covers_public_general_and_unsupported_answers():
    suite = load_suite(FAQ_DATASET)
    assert len(suite.cases) == 4
    assert sum(len(case.turns) for case in suite.cases) == 6
    assert all(case.review_goal for case in suite.cases)
    assert {case.id for case in suite.cases} == {
        "law_followup", "care_guidance", "demo_scope", "personal_eligibility",
    }
    for case in suite.cases:
        for turn in case.turns:
            assert turn.expected is not None
            if case.id == "demo_scope":
                assert turn.expected.intent == "general"
                assert "main_agent_response" in turn.expected.nodes_include
                assert "knowledge_search" in turn.expected.tools_exclude
            else:
                assert turn.expected.intent == "faq"
                assert "knowledge_search" in turn.expected.tools_include


def test_model_reviewer_requires_real_reply_and_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate(
        {
            "name": "review",
            "cases": [
                {
                    "id": "one",
                    "description": "预约",
                    "review_goal": "用户完成预约对话",
                    "user_id": "U1001",
                    "turns": [
                        {
                            "id": "start",
                            "message": "我想预约",
                            "expected": {"intent": "appointment"},
                        }
                    ],
                }
            ],
        }
    )

    class PassingModel:
        def invoke(self, messages):
            assert "用户完成预约对话" in messages[1].content
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "verdict": "pass",
                        "task_completed": True,
                        "reason": "对话目标完成",
                        "issues": [],
                    },
                    ensure_ascii=False,
                )
            )

    report = _run_suite(
        suite, mode="offline", reviewer=ConversationReviewer(PassingModel()), review="model"
    )
    assert report["summary"]["structural_passed"]  # 仅结构断言通过
    assert not report["summary"]["review_passed"]
    assert not report["summary"]["overall_passed"]
    assert report["cases"][0]["review"]["status"] == "failed"
    assert "未生成模型回复" in report["cases"][0]["review"]["reason"]
    _, xlsx_path = save_report(report, tmp_path / "reports")
    with zipfile.ZipFile(xlsx_path) as workbook:
        assert "对话评审" in workbook.read("xl/workbook.xml").decode("utf-8")
        assert "未生成模型回复" in workbook.read("xl/worksheets/sheet3.xml").decode("utf-8")

    class InventedEvidenceModel:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "verdict": "fail",
                        "task_completed": False,
                        "reason": "没有完成",
                        "issues": [
                            {
                                "turn_id": "start",
                                "category": "reply",
                                "evidence_quote": "不存在的原文",
                                "reason": "错误",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    invalid = ConversationReviewer(InventedEvidenceModel()).review(
        goal="完成预约", turns=report["cases"][0]["turns"]
    )
    assert invalid["status"] == "unavailable"
    assert invalid["error_type"] == "ValueError"

    synthetic_turns = [
        {
            "turn_id": "t1",
            "message": "查询预约",
            "answer": "已查到预约记录。",
            "response_info": {"mode": "llm"},
            "node_path": ["supervisor"],
            "tool_trace": [{"tool": "query_user_appointments"}],
            "result": {"status": "appointments_found"},
        }
    ]
    accepted = ConversationReviewer(PassingModel()).review(
        goal="用户完成预约对话", turns=synthetic_turns
    )
    assert accepted["status"] == "passed"
    assert accepted["provider"] == "direct"

    class GroundedFailureModel:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "verdict": "fail",
                        "task_completed": False,
                        "reason": "回复没有完成查询目标",
                        "issues": [
                            {
                                "turn_id": "t1",
                                "category": "reply",
                                "evidence_quote": "查询预约",
                                "reason": "用户询问预约记录",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    rejected = ConversationReviewer(GroundedFailureModel()).review(
        goal="用户获知预约详情", turns=synthetic_turns
    )
    assert rejected["status"] == "failed"
    assert rejected["issues"][0]["turn_id"] == "t1"


def test_node_assertion_is_checked_against_real_path(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate(
        {
            "name": "nodes",
            "cases": [
                {
                    "id": "one",
                    "user_id": "U1001",
                    "turns": [
                        {
                            "id": "query",
                            "message": "查询我的预约记录",
                            "expected": {
                                "nodes_include": ["appointment_subgraph"],
                                "nodes_exclude": ["nonexistent_node"],
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert _run_suite(suite, mode="offline", review="none")["summary"]["structural_passed"]
    suite.cases[0].turns[0].expected.nodes_include = ["made_up_node"]
    bad = _run_suite(suite, mode="offline", review="none")
    assert "缺少节点 made_up_node" in bad["cases"][0]["turns"][0]["failures"]


def test_excel_example_matches_json_and_runs_without_model(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    json_suite = load_suite(DATASET)
    excel_suite = load_suite(EXCEL_DATASET)
    assert excel_suite.model_dump(exclude={"name", "version"}) == json_suite.model_dump(
        exclude={"name", "version"}
    )
    previews = [(case.id, turn.id, turn.expected) for case in excel_suite.cases
                for turn in case.turns if turn.id in {"site", "cancel_request", "checkin_request"}]
    assert len(previews) == 3
    assert all(expected.reply_mode == "llm" and expected.preview_confirmable is True
               for _, _, expected in previews)
    assert not _run_suite(excel_suite, mode="offline", review="none")["summary"][
        "structural_passed"
    ]


def test_unavailable_preview_is_valid_offline_diagnostic_but_not_model_acceptance(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate({
        "name": "preview-unavailable", "cases": [{
            "id": "booking", "user_id": "U1001", "turns": [
                {"id": "preview", "message": "预约明天上午到市中心献血屋", "expected": {
                    "result_status": "preview_unavailable", "reply_mode": "unavailable",
                    "preview_confirmable": False, "tools_exclude": ["submit_appointment"],
                }},
                {"id": "confirm", "message": "确认预约", "expected": {
                    "result_status": "preview_unavailable", "reply_mode": "unavailable",
                    "preview_confirmable": False, "tools_exclude": ["submit_appointment"],
                }},
            ],
        }],
    })
    report = _run_suite(suite, mode="offline", review="none")
    assert report["summary"]["structural_passed"]
    assert not report["summary"]["passed"]
    assert report["summary"]["llm_replies"] == 0
    assert all(turn["passed"] for turn in report["cases"][0]["turns"])


def test_excel_rejects_formula_and_invalid_turn_references(tmp_path):
    tables = {
        "场景": [INPUT_HEADERS["场景"], ["case", "说明", "U1001", "否"]],
        "对话": [INPUT_HEADERS["对话"], ["case", "turn", "你好"]],
        "断言": [INPUT_HEADERS["断言"], ["case", "missing", "意图", "faq"]],
    }
    with pytest.raises(ValueError, match="未找到对应对话轮次"):
        suite_data_from_tables(tables, "bad")
    tables["断言"][1][1] = "turn"
    tables["断言"].append(["case", "turn", "可确认预览", "否"])
    assert suite_data_from_tables(tables, "diagnostic")["cases"][0]["turns"][0][
        "expected"
    ]["preview_confirmable"] is False
    source = write_workbook(
        tmp_path / "formula.xlsx",
        [(name, rows, [25] * len(rows[0]), None) for name, rows in tables.items()],
    )
    with zipfile.ZipFile(source) as archive:
        payloads = {entry.filename: archive.read(entry.filename) for entry in archive.infolist()}
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ET.fromstring(payloads["xl/worksheets/sheet2.xml"])
    first_cell = root.find(f".//{{{namespace}}}sheetData/{{{namespace}}}row[2]/{{{namespace}}}c[3]")
    assert first_cell is not None
    ET.SubElement(first_cell, f"{{{namespace}}}f").text = "1+1"
    payloads["xl/worksheets/sheet2.xml"] = ET.tostring(root, encoding="utf-8")
    edited = tmp_path / "edited.xlsx"
    with zipfile.ZipFile(edited, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="不允许公式"):
        read_input_tables(edited)


def test_wrong_expectation_fails_and_dialogue_without_expectation_is_ungraded(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate(
        {
            "name": "negative",
            "cases": [
                {
                    "id": "case",
                    "user_id": "U1001",
                    "turns": [
                        {
                            "id": "wrong",
                            "message": "查询我的预约记录",
                            "expected": {"result_status": "appointments_found"},
                        },
                        {"id": "observe", "message": "你好"},
                    ],
                }
            ],
        }
    )
    report = _run_suite(suite, mode="offline", review="none")
    first, second = report["cases"][0]["turns"]
    assert first["verdict"] == "failed"
    assert any("result_status" in failure for failure in first["failures"])
    assert second["verdict"] == "ungraded"
    assert not report["summary"]["passed"]
    assert report["summary"]["turns_ungraded"] == 1


def test_model_mode_cannot_count_rule_fallback_as_model_success(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate(
        {
            "name": "model-gate",
            "cases": [
                {
                    "id": "case",
                    "user_id": "U1001",
                    "turns": [
                        {
                            "id": "start",
                            "message": "我要预约",
                            "expected": {"intent": "appointment"},
                        },
                    ],
                }
            ],
        }
    )
    report = run_suite(
        suite,
        router_factory=IntentRouter,
        reviewer=ConversationReviewer(None),
    )
    failures = report["cases"][0]["turns"][0]["failures"]
    assert any("未生成回复" in failure for failure in failures)
    assert any("理解未走模型" in failure for failure in failures)
    assert report["summary"]["llm_replies"] == 0
    assert not report["summary"]["passed"]


def test_model_gate_applies_even_without_explicit_assertion(monkeypatch):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    suite = ConversationSuite.model_validate(
        {
            "name": "model-only",
            "cases": [{"id": "one", "user_id": "U1001", "turns": [
                {"id": "hello", "message": "你好"}
            ]}],
        }
    )
    report = run_suite(
        suite, router_factory=IntentRouter, reviewer=ConversationReviewer(None)
    )
    turn = report["cases"][0]["turns"][0]
    assert turn["verdict"] == "failed"
    assert any("未生成回复" in item for item in turn["failures"])
    assert any("navigation 理解未走模型" in item for item in turn["failures"])
    assert report["cases"][0]["review"]["status"] == "unavailable"
    assert not report["summary"]["overall_passed"]


def test_no_business_assertion_requires_model_gate_and_judge():
    turn = ConversationSuite.model_validate({
        "name": "judge-only", "cases": [{"id": "one", "user_id": "U1001", "turns": [
            {"id": "hello", "message": "你好"}
        ]}]
    }).cases[0].turns[0]
    state = {"messages": [SimpleNamespace(content="你好，有什么可以帮您？")],
             "response_info": {"mode": "llm"}}
    record = _turn_record(turn, state, [], 1.0, mode="model")
    assert record["verdict"] == "model_checked"
    assert record["passed"]
    assert record["expected"] is None


def test_progress_heartbeat_during_slow_model_call():
    from threading import Event

    messages = []
    with _heartbeat(messages.append, "评审 ", interval=0.01):
        Event().wait(0.04)
    assert any("仍在执行" in message for message in messages)


def test_invalid_suite_rejected_before_execution(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "name": "bad",
                "cases": [
                    {
                        "id": "case",
                        "user_id": "U1001",
                        "turns": [
                            {"id": "same", "message": "你好", "expected": {"made_up_field": 1}},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_suite(bad)
    with pytest.raises(ValidationError):
        ConversationSuite.model_validate(
            {
                "name": "empty-assertion",
                "cases": [
                    {
                        "id": "c",
                        "user_id": "U1001",
                        "turns": [
                            {"id": "t", "message": "你好", "expected": {}},
                        ],
                    }
                ],
            }
        )
    assert list(tmp_path.iterdir()) == [bad]


def test_cli_creates_report_and_streams_progress(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLOOD_AGENT_TRACE_MODE", "off")
    monkeypatch.setattr(IntentRouter, "from_env", classmethod(lambda cls: IntentRouter()))
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "name": "capture",
                "cases": [
                    {
                        "id": "c",
                        "user_id": "U1001",
                        "turns": [
                            {"id": "t", "message": "你好"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--dataset", str(path), "--output-dir", str(tmp_path / "reports")]) == 1
    progress = capsys.readouterr().err
    assert "读取用例" in progress
    assert "场景 1/1 [c] 开始" in progress
    assert "第 1/1 轮 [t] 开始" in progress
    assert "模型评审结束" in progress
    assert "报告已保存" in progress
    assert "本次综合判定：未通过" in progress
    assert len(list((tmp_path / "reports").glob("*.json"))) == 1
    assert len(list((tmp_path / "reports").glob("*.xlsx"))) == 1


def test_cli_rejects_disabling_either_model_chain():
    with pytest.raises(SystemExit) as offline:
        main(["--dataset", str(DATASET), "--mode", "offline"])
    assert offline.value.code == 2
    with pytest.raises(SystemExit) as no_review:
        main(["--dataset", str(DATASET), "--review", "none"])
    assert no_review.value.code == 2
