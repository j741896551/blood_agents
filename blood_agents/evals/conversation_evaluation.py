"""多轮对话回归：逐轮运行真实 LangGraph，按可核验业务事实自动判定。

固定话术来自 Excel/JSON；每个场景独立 SQLite + MemorySaver。结构断言和
模型执行门禁和模型对话评审各自给结论；正式验收不能关闭任一模型链路。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field, model_validator

from blood_agents.log.trace_logging import mask_sensitive
from blood_agents.orchestration.conversation import ConversationGraph
from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter

from .conversation_reviewer import ConversationReviewer
from .excel_io import read_input_tables, suite_data_from_tables, write_report_xlsx

Progress = Callable[[str], None]

# 固定话术可引用同一场景前一轮 Mock 回执中的随机编号；只开放必要字段，
# 避免测试脚本从任意模型文本或其他用户的结果中拼接业务参数。
RECEIPT_REFERENCE = re.compile(r"\{\{(address_id|redemption_id):([A-Za-z0-9_-]+)\}\}")


def _progress(progress: Progress | None, message: str) -> None:
    """进度只包含场景/轮次 ID 与脱敏状态，详细输入输出仍由追踪日志负责。"""
    if progress is not None:
        progress(message)


@contextmanager
def _heartbeat(progress: Progress | None, label: str, *, interval: float = 10.0):
    """阻塞等待模型时定期报进度，避免长超时被误认为程序卡住。"""
    if progress is None:
        yield
        return
    stopped = Event()
    started = monotonic()

    def announce_wait() -> None:
        """在评测等待模型响应时输出当前进度。"""
        while not stopped.wait(interval):
            _progress(progress, f"{label}仍在执行，已等待 {monotonic() - started:.0f} 秒")

    worker = Thread(target=announce_wait, name="conversation-eval-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=1.0)


class TurnExpected(BaseModel):
    """只比较明确声明的字段；未声明的自然语言内容不被假装为自动验收。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    intent: Optional[str] = None
    action: Optional[str] = None
    result_status: Optional[str] = None
    result_code: Optional[str] = None
    result_success: Optional[bool] = None
    operation: Optional[str] = None
    data_status: Optional[str] = None
    appointment_statuses: Optional[list[str]] = None
    agent_status: Optional[str] = None
    active_agent: Optional[str] = None
    slots: dict[str, Any] = Field(default_factory=dict)
    tools_exact: Optional[list[str]] = None
    tools_include: list[str] = Field(default_factory=list)
    tools_exclude: list[str] = Field(default_factory=list)
    nodes_exact: Optional[list[str]] = None
    nodes_include: list[str] = Field(default_factory=list)
    nodes_exclude: list[str] = Field(default_factory=list)
    same_appointment_id_as: Optional[str] = None
    reply_mode: Optional[str] = None
    # 只表示图状态允许下一轮确认，并不证明浏览器已实际展示这条回复。
    preview_confirmable: Optional[bool] = None
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_effective_assertion(self):
        """判断测试轮次是否配置了有效断言。"""
        if not any(getattr(self, name) not in ({}, []) for name in self.model_fields_set):
            raise ValueError("expected 不能是空断言；只记录结果时请省略 expected")
        return self


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # 可以只提供话术先采集轨迹；未写断言的轮次只能标“待评”，不能算通过。
    expected: Optional[TurnExpected] = None


class ConversationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    description: str = ""
    review_goal: str = ""
    user_id: str = Field(min_length=1)
    seed_appointment_samples: bool = False
    turns: list[ConversationTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_turn_ids(self):
        """校验并提取场景内唯一的轮次编号。"""
        ids = [turn.id for turn in self.turns]
        if len(ids) != len(set(ids)):
            raise ValueError(f"场景 {self.id} 的 turn.id 不能重复")
        return self


class ConversationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = 1
    name: str = Field(min_length=1)
    cases: list[ConversationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self):
        """校验并提取批量评测中的唯一场景编号。"""
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("场景 id 不能重复")
        return self


def load_suite(path: str | Path) -> ConversationSuite:
    """先校验完整输入，再创建任何输出文件或触发模型/工具。"""
    source = Path(path)
    if source.suffix.lower() == ".xlsx":
        data = suite_data_from_tables(read_input_tables(source), source.stem)
        return ConversationSuite.model_validate(data)
    if source.suffix.lower() == ".json":
        return ConversationSuite.model_validate_json(source.read_text(encoding="utf-8"))
    raise ValueError("用例文件须为 .xlsx 或 .json")


def _resolve_receipt_references(message: str, prior_results: dict[str, dict]) -> str:
    """把 {{address_id:轮次ID}} 等占位符替换为此前已执行的工具回执字段。"""
    def replace(match: re.Match[str]) -> str:
        """替换评测模板中的参数占位符。"""
        field, turn_id = match.groups()
        result = prior_results.get(turn_id)
        if result is None:
            raise ValueError(f"引用的轮次 {turn_id!r} 尚无执行结果")
        data = result.get("data") or {}
        value = data.get(field) if isinstance(data, dict) else None
        if not isinstance(value, str) or not value:
            raise ValueError(f"轮次 {turn_id!r} 的 Mock 回执没有 {field}")
        return value

    return RECEIPT_REFERENCE.sub(replace, message)


def _appointment_ids(result: dict) -> set[str]:
    """从评测结果中提取可用于断言的预约编号。"""
    ids = set()
    data = result.get("data") or {}
    if isinstance(data, dict) and data.get("appointment_id"):
        ids.add(data["appointment_id"])
    if (result.get("appointment") or {}).get("appointment_id"):
        ids.add(result["appointment"]["appointment_id"])
    ids.update(
        item["appointment_id"]
        for item in result.get("appointments") or []
        if item.get("appointment_id")
    )
    return ids


def _check_turn(
    expected: TurnExpected, state: dict, *, prior_ids: dict[str, set[str]]
) -> list[str]:
    """结构化业务断言是主裁判；自然语言仅用用户给出的包含/排除断言。"""
    result = state.get("result") or {}
    tools = [item["tool"] for item in state.get("tool_trace") or []]
    nodes = [item.get("node") for item in state.get("node_trace") or []]
    answer = str(state.get("messages", [])[-1].content) if state.get("messages") else ""
    actual = {
        "intent": state.get("intent"),
        "action": (state.get("understanding") or {}).get("action"),
        "result_status": result.get("status"),
        "result_code": result.get("code"),
        "result_success": result.get("success"),
        "operation": result.get("operation"),
        "data_status": (result.get("data") or {}).get("status"),
        "agent_status": state.get("agent_status"),
        "active_agent": state.get("active_agent"),
        "reply_mode": (state.get("response_info") or {}).get("mode"),
        "preview_confirmable": bool((state.get("slots") or {}).get("preview_delivered")),
    }
    failures = []
    for field, found in actual.items():
        wanted = getattr(expected, field)
        if field in expected.model_fields_set and found != wanted:
            failures.append(f"{field}: 预期 {wanted!r}，实际 {found!r}")
    for key, wanted in expected.slots.items():
        found = (state.get("slots") or {}).get(key)
        if found != wanted:
            failures.append(f"slots.{key}: 预期 {wanted!r}，实际 {found!r}")
    if expected.appointment_statuses is not None:
        found = sorted(item.get("status") for item in result.get("appointments") or [])
        wanted = sorted(expected.appointment_statuses)
        if found != wanted:
            failures.append(f"appointment_statuses: 预期 {wanted!r}，实际 {found!r}")
    if expected.tools_exact is not None and tools != expected.tools_exact:
        failures.append(f"tools_exact: 预期 {expected.tools_exact!r}，实际 {tools!r}")
    for name in expected.tools_include:
        if name not in tools:
            failures.append(f"缺少工具 {name}")
    for name in expected.tools_exclude:
        if name in tools:
            failures.append(f"不应调用工具 {name}")
    if expected.nodes_exact is not None and nodes != expected.nodes_exact:
        failures.append(f"nodes_exact: 预期 {expected.nodes_exact!r}，实际 {nodes!r}")
    for name in expected.nodes_include:
        if name not in nodes:
            failures.append(f"缺少节点 {name}")
    for name in expected.nodes_exclude:
        if name in nodes:
            failures.append(f"不应经过节点 {name}")
    if expected.same_appointment_id_as is not None:
        prior = prior_ids.get(expected.same_appointment_id_as)
        current = _appointment_ids(result)
        if not prior:
            failures.append(f"引用的上一轮 {expected.same_appointment_id_as!r} 没有预约号")
        elif not prior & current:
            failures.append(f"本轮预约号 {sorted(current)!r} 与上一轮 {sorted(prior)!r} 不一致")
    for phrase in expected.answer_contains:
        if phrase not in answer:
            failures.append(f"回复缺少文字 {phrase!r}")
    for phrase in expected.answer_not_contains:
        if phrase in answer:
            failures.append(f"回复不应包含文字 {phrase!r}")
    return failures


def _check_model_execution(state: dict) -> list[str]:
    """每一轮都检查模型路径，即使用户没有填写显式业务断言。"""
    failures = []
    reply_mode = (state.get("response_info") or {}).get("mode")
    if reply_mode != "llm":
        failures.append(f"真实模型模式未生成回复：reply_mode={reply_mode!r}")
    model_info = state.get("model_info") or {}
    stages = ["navigation"]
    if state.get("intent") == "appointment":
        stages.append("appointment")
        action = (state.get("understanding") or {}).get("action", "collect")
        if action not in {"cancel", "checkin"}:
            found = (state.get("tool_plan_info") or {}).get("mode")
            if found != "llm":
                failures.append(f"真实模型模式的 appointment_tool_plan 未走模型：mode={found!r}")
    elif state.get("intent") == "points":
        stages.append("points")
    for stage in stages:
        found = (model_info.get(stage) or {}).get("mode")
        if found != "llm":
            failures.append(f"真实模型模式的 {stage} 理解未走模型：mode={found!r}")
    return failures


def _turn_record(
    turn: ConversationTurn, state: dict, failures: list[str], elapsed_ms: float, *, mode: str
) -> dict:
    """把一次对话执行结果整理为评测记录。"""
    result = state.get("result") or {}
    message = state.get("messages") or []
    return mask_sensitive(
        {
            "turn_id": turn.id,
            "message": turn.message,
            "verdict": ("failed" if failures else "model_checked" if mode == "model" else "ungraded")
            if turn.expected is None
            else ("failed" if failures else "passed"),
            "passed": (turn.expected is not None or mode == "model") and not failures,
            "failures": failures,
            "expected": turn.expected.model_dump(exclude_unset=True) if turn.expected else None,
            "answer": message[-1].content if message else "",
            "trace_id": state.get("trace_id"),
            "intent": state.get("intent"),
            "understanding": state.get("understanding"),
            "slots": state.get("slots"),
            "result": result,
            # FAQ 评审需要看到真正交给回复模型的已核验片段；仅凭来源列表
            # 无法判断最终答复是否添加了片段之外的事实。
            "retrieved_context": state.get("retrieved_context") or []
            if state.get("intent") == "faq"
            else [],
            "model_info": state.get("model_info"),
            "response_info": state.get("response_info"),
            "reply_generated_by_llm": (state.get("response_info") or {}).get("mode") == "llm",
            "agent_handoff": state.get("agent_handoff"),
            "node_path": [item.get("node") for item in state.get("node_trace") or []],
            "tool_trace": state.get("tool_trace") or [],
            "elapsed_ms": round(elapsed_ms, 2),
        }
    )


def _run_suite(
    suite: ConversationSuite,
    *,
    mode: str,
    router_factory=None,
    review: str,
    reviewer: ConversationReviewer | None = None,
    progress: Progress | None = None,
) -> dict:
    """按场景隔离业务替身与会话，运行固定多轮话术并核验执行证据。"""
    # 评测数据提供方是可选本地插件；Agent 核心包不包含合成业务实现。
    try:
        from local_business.api import MockLegacyBloodAPI
        from local_business.repository import BloodRepository
    except ImportError as exc:
        raise RuntimeError("批量业务评测需要单独安装本地业务替身插件") from exc
    if mode not in {"offline", "model"}:
        raise ValueError("mode 必须是 offline 或 model")
    if review not in {"none", "model"}:
        raise ValueError("review 必须是 none 或 model")
    if review == "model" and reviewer is None:
        reviewer = ConversationReviewer(IntentRouter.from_env().response_llm)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    _progress(progress, f"运行 {run_id} 开始：{len(suite.cases)} 个场景，"
              f"{sum(len(case.turns) for case in suite.cases)} 轮对话")
    records = []
    for case_index, case in enumerate(suite.cases, start=1):
        _progress(progress, f"场景 {case_index}/{len(suite.cases)} [{case.id}] 开始："
                  f"用户 {case.user_id}，{len(case.turns)} 轮")
        with tempfile.TemporaryDirectory(prefix="blood-conversation-eval-") as temporary:
            repo = BloodRepository(str(Path(temporary) / "case.db"))
            repo.seed_demo_data(include_appointment_samples=case.seed_appointment_samples)
            api = MockLegacyBloodAPI(repo)
            router = (
                router_factory()
                if router_factory
                else (IntentRouter() if mode == "offline" else IntentRouter.from_env())
            )
            graph = ConversationGraph(
                build_graph(
                    repo,
                    router,
                    legacy_api=api,
                    checkpointer=MemorySaver(),
                )
            )
            session_id = f"evaluation-{run_id}-{case.id}"
            prior_ids: dict[str, set[str]] = {}
            prior_results: dict[str, dict] = {}
            turns = []
            for index, turn in enumerate(case.turns):
                label = f"场景 [{case.id}] 第 {index + 1}/{len(case.turns)} 轮 [{turn.id}] "
                _progress(progress, f"{label}开始，正在调用智能体")
                started = monotonic()
                try:
                    resolved_turn = turn.model_copy(update={
                        "message": _resolve_receipt_references(turn.message, prior_results)
                    })
                    with _heartbeat(progress, label):
                        state = graph.invoke(
                            {"messages": [HumanMessage(content=resolved_turn.message)],
                             "user_id": case.user_id},
                            config={
                                "configurable": {"thread_id": session_id},
                                "metadata": {"entrypoint": "conversation_evaluation"},
                            },
                        )
                    failures = _check_model_execution(state) if mode == "model" else []
                    if turn.expected:
                        failures.extend(_check_turn(turn.expected, state, prior_ids=prior_ids))
                    prior_ids[turn.id] = _appointment_ids(state.get("result") or {})
                    prior_results[turn.id] = state.get("result") or {}
                    record = _turn_record(
                        resolved_turn, state, failures, (monotonic() - started) * 1000, mode=mode
                    )
                    turns.append(record)
                    navigation = (record.get("model_info") or {}).get("navigation") or {}
                    response = record.get("response_info") or {}
                    appointment = (record.get("model_info") or {}).get("appointment") or {}
                    plan = state.get("tool_plan_info") or {}
                    appointment_modes = (
                        f"；预约理解={appointment.get('mode')}；工具规划={plan.get('mode')}"
                        if record.get("intent") == "appointment" else ""
                    )
                    _progress(
                        progress,
                        f"{label}结束：{record['verdict']}；意图={record.get('intent')}；"
                        f"导航模型={navigation.get('mode')}；回复模型={response.get('mode')}；"
                        f"耗时={record['elapsed_ms'] / 1000:.1f} 秒；"
                        f"trace_id={record.get('trace_id')}"
                        + appointment_modes
                        + (f"；首个问题={failures[0]}" if failures else ""),
                    )
                except Exception as exc:  # noqa: BLE001 - 用例异常必须转成报告，不能吞掉
                    _progress(progress, f"{label}异常：{type(exc).__name__}；后续依赖轮次跳过")
                    turns.append(
                        mask_sensitive(
                            {
                                "turn_id": turn.id,
                                "message": turn.message,
                                "verdict": "failed",
                                "passed": False,
                                "failures": [f"执行异常：{type(exc).__name__}: {exc}"],
                                "expected": turn.expected.model_dump(exclude_unset=True)
                                if turn.expected
                                else None,
                                "elapsed_ms": round((monotonic() - started) * 1000, 2),
                            }
                        )
                    )
                    for skipped in case.turns[index + 1 :]:
                        turns.append(
                            mask_sensitive(
                                {
                                    "turn_id": skipped.id,
                                    "message": skipped.message,
                                    "verdict": "skipped",
                                    "passed": False,
                                    "failures": ["前一轮执行异常，依赖会话状态未执行"],
                                    "expected": (
                                        skipped.expected.model_dump(exclude_unset=True)
                                        if skipped.expected
                                        else None
                                    ),
                                }
                            )
                        )
                    break  # 同场景后续轮次依赖前一轮 checkpoint，不能继续误判。
            if review == "model":
                _progress(progress, f"场景 [{case.id}] 开始模型评审")
                with _heartbeat(progress, f"场景 [{case.id}] 模型评审 "):
                    case_review = reviewer.review(
                        goal=case.review_goal or case.description, turns=turns
                    )
                _progress(progress, f"场景 [{case.id}] 模型评审结束："
                          f"{case_review['status']}；服务={case_review.get('provider') or '无'}")
            else:
                case_review = {
                    "status": "not_requested",
                    "reason": "未启用模型评审",
                    "issues": [],
                    "provider": None,
                }
            records.append(
                {
                    "case_id": case.id,
                    "description": case.description,
                    "review_goal": case.review_goal or case.description,
                    "user_id": case.user_id,
                    "session_id": session_id,
                    "passed": len(turns) == len(case.turns)
                    and all(item["passed"] for item in turns),
                    "turns": turns,
                    "review": case_review,
                }
            )
            _progress(progress, f"场景 [{case.id}] 完成："
                      f"结构/模型门禁={'通过' if records[-1]['passed'] else '未通过'}；"
                      f"模型评审={case_review['status']}")
    total_turns = sum(len(item["turns"]) for item in records)
    passed_turns = sum(turn["passed"] for item in records for turn in item["turns"])
    ungraded_turns = sum(
        turn["verdict"] == "ungraded" for item in records for turn in item["turns"]
    )
    model_only_turns = sum(
        turn["verdict"] == "model_checked" for item in records for turn in item["turns"]
    )
    llm_replies = sum(
        turn.get("reply_generated_by_llm", False) for item in records for turn in item["turns"]
    )
    review_passed = (
        all(item["review"]["status"] == "passed" for item in records) if review == "model" else None
    )
    structural_passed = all(item["passed"] for item in records)
    return {
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "suite": suite.name,
        "mode": mode,
        "review_mode": review,
        "evaluation_scope": (
            "model_conversation_and_mock_business"
            if mode == "model"
            else "orchestration_and_mock_business_only"
        ),
        "suite_sha256": hashlib.sha256(suite.model_dump_json().encode("utf-8")).hexdigest(),
        "environment": {
            "business_api": "local_mock",
            "database": "isolated_temp_sqlite",
            "model": "configured_provider" if mode == "model" else "disabled_rules_only",
            "python": platform.python_version(),
        },
        "summary": {
            "cases_total": len(records),
            "cases_passed": sum(item["passed"] for item in records),
            "turns_total": total_turns,
            "turns_passed": passed_turns,
            "turns_ungraded": ungraded_turns,
            "turns_model_only": model_only_turns,
            "llm_replies": llm_replies,
            "passed": structural_passed and review_passed if review == "model" else False,
            "structural_passed": structural_passed,
            "review_cases_passed": sum(item["review"]["status"] == "passed" for item in records),
            "review_passed": review_passed,
            "overall_passed": structural_passed and review_passed if review == "model" else False,
        },
        "cases": records,
        "limitations": [
            "自然度与未声明的语义预期需要人工复核；自动分数仅来自显式断言。",
            "离线模式只验证规则与 Mock 编排，不代表真实模型质量。",
            "本地 Mock 成功不代表公司旧接口或生产环境通过。",
        ],
    }


def run_suite(
    suite: ConversationSuite,
    *,
    router_factory=None,
    reviewer: ConversationReviewer | None = None,
    progress: Progress | None = None,
) -> dict:
    """正式对话验收：被测智能体与评审必须都实际走模型。"""
    return _run_suite(
        suite, mode="model", review="model", router_factory=router_factory,
        reviewer=reviewer, progress=progress
    )


def save_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """JSON 留机器追溯，Excel 供人阅读；不再生成 Markdown。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{report['run_id']}.json"
    xlsx_path = directory / f"{report['run_id']}.xlsx"
    descriptor = os.open(json_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
    write_report_xlsx(report, xlsx_path)
    return json_path, xlsx_path


def main(argv: list[str] | None = None) -> int:
    """读取批量用例并执行多轮对话评测。"""
    parser = argparse.ArgumentParser(description="逐轮运行智慧献血多智能体对话用例")
    parser.add_argument("--dataset", required=True, help="多轮对话 Excel(.xlsx) 或 JSON 用例文件")
    parser.add_argument("--output-dir", default="evals/reports", help="脱敏 Excel/JSON 报告目录")
    args = parser.parse_args(argv)
    suite = load_suite(args.dataset)
    def show_progress(message: str) -> None:
        """向终端展示评测的阶段进度。"""
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", file=sys.stderr, flush=True)

    show_progress(f"读取用例：{Path(args.dataset).resolve()}")
    show_progress(f"报告目录：{Path(args.output_dir).resolve()}")
    report = run_suite(suite, progress=show_progress)
    json_path, xlsx_path = save_report(report, args.output_dir)
    show_progress(f"报告已保存：Excel={xlsx_path.resolve()}；JSON={json_path.resolve()}")
    show_progress(f"本次综合判定：{'通过' if report['summary']['overall_passed'] else '未通过'}")
    print(
        json.dumps(
            {"summary": report["summary"], "json": str(json_path), "xlsx": str(xlsx_path)},
            ensure_ascii=False,
        )
    )
    return 0 if report["summary"]["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
