"""对 RAG 挑战集运行模型证据核验；模型故障单独计数，不充作安全拒答。"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from blood_agents.evals.rag_comparison import dataset_identity, load_cases
from blood_agents.knowledge.evidence import EvidenceAssessor, EvidenceCheckUnavailable
from blood_agents.knowledge.rag import default_retriever
from blood_agents.orchestration.router import IntentRouter


def select_cases(cases: list[dict], case_ids: list[str] | None) -> list[dict]:
    """定向复测仍读取完整数据集，只筛选执行题目，不改写原始版本指纹。"""
    if not case_ids:
        return cases
    requested = set(case_ids)
    available = {case["id"] for case in cases}
    unknown = requested - available
    if unknown:
        raise ValueError(f"数据集不存在题目：{', '.join(sorted(unknown))}")
    return [case for case in cases if case["id"] in requested]


def save_report(report: dict, output: Path) -> None:
    """模型评测可能包含自定义用户话术；报告只允许当前用户读取且不覆盖旧报告。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def evaluate_evidence(cases: list[dict], retriever, assessor: EvidenceAssessor,
                      top_k: int = 4, on_case=None) -> dict:
    """正例要求选中人工标注来源；负例要求明确不支持，核验错误不能算通过。"""
    if not cases or top_k <= 0:
        raise ValueError("证据核验评测需要非空用例和正整数 top_k")
    rows = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        status = "checked"
        info = {"mode": "skipped"}
        hits = []
        selected = []
        try:
            hits = retriever.search(case["query"], top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - 评测报告需记录任意检索故障
            status = "retrieval_failed"
            info = {"mode": "unavailable", "error_type": type(exc).__name__}
        else:
            if hits:
                try:
                    selected, info = assessor.assess(case["query"], hits)
                except EvidenceCheckUnavailable as exc:
                    status = "check_failed"
                    info = exc.info
                except Exception as exc:  # noqa: BLE001 - 意外的核验故障不可算安全拒答
                    status = "check_failed"
                    info = {"mode": "unavailable", "error_type": type(exc).__name__}
        sources = list(dict.fromkeys(hit["source"] for hit in selected))
        expected = case["sources"]
        passed = (status == "checked" and
                  (bool(set(sources) & set(expected)) if expected else not selected))
        row = {
            "id": case["id"], "group": case["group"], "query": case["query"],
            "expected_sources": expected,
            "retrieval_hit_count": len(hits),
            "selected_doc_ids": [hit["doc_id"] for hit in selected],
            "selected_sources": sources, "status": status, "passed": passed,
            "model_info": info,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        rows.append(row)
        if on_case:
            on_case(index, len(cases), row)
    positives = [row for row in rows if row["expected_sources"]]
    negatives = [row for row in rows if not row["expected_sources"]]
    return {
        "metrics": {
            "count": len(rows), "positive_count": len(positives),
            "negative_count": len(negatives),
            "positive_source_supported_rate": round(
                sum(row["passed"] for row in positives) / len(positives), 4,
            ) if positives else None,
            "negative_rejection_rate": round(
                sum(row["passed"] for row in negatives) / len(negatives), 4,
            ) if negatives else None,
            "check_failed_count": sum(row["status"] == "check_failed" for row in rows),
            "retrieval_failed_count": sum(row["status"] == "retrieval_failed" for row in rows),
        },
        "cases": rows,
        "limitations": [
            "人工标注核对来源和应否回答，不验证最终自然语言回复质量或医学事实正确性。",
            "模型核验可能误放行或误拒答；结构通过不等于真实模型性能通过。",
            "查询与片段会发送至配置的模型服务；本挑战集仅含公开摘要、虚构演示文档和合成问题。",
        ],
    }


def main() -> None:
    """运行 RAG 证据核验样本并保存报告。"""
    parser = argparse.ArgumentParser(description="用已配置模型评测 FAQ 证据充分性")
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).parents[2] / "evals" / "rag_challenge.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--case-id", action="append", help="仅复测指定题目；可重复使用")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    all_cases = load_cases(args.dataset)
    try:
        cases = select_cases(all_cases, args.case_id)
    except ValueError as exc:
        parser.error(str(exc))
    router = IntentRouter.from_env()
    if not router.evidence_llm:
        parser.error("必须先配置可用的大模型；本评测不使用规则或假模型代替证据核验")
    retriever = default_retriever()

    def progress(index, total, row):
        """输出当前批量评测的处理进度。"""
        print(f"[{index}/{total}] {row['id']} 证据核验={row['status']} "
              f"判定={'通过' if row['passed'] else '未通过'}", flush=True)

    report = evaluate_evidence(cases, retriever, EvidenceAssessor(router.evidence_llm),
                               top_k=args.top_k, on_case=progress)
    report.update({"backend": retriever.backend, "top_k": args.top_k,
                   "dataset_total_cases": len(all_cases),
                   "selected_case_ids": [case["id"] for case in cases],
                   "dataset": str(args.dataset.resolve()),
                   "created_at": datetime.now(timezone.utc).isoformat()})
    report.update(dataset_identity(args.dataset))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output or (Path(__file__).parents[2] / "evals" / "reports"
                             / f"rag_evidence_{run_id}.json")
    save_report(report, output)
    print(json.dumps({"report": str(output), "metrics": report["metrics"],
                      "failed_cases": [row["id"] for row in report["cases"]
                                       if not row["passed"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
