"""同一批静态知识问题，对比原检索与 bge-reranker 精排后的结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from blood_agents.knowledge.adapters import BGEReranker
from blood_agents.knowledge.rag import HybridRetriever, default_retriever, load_approved_corpus


def dataset_identity(path: Path) -> dict:
    """报告保留版本和文件指纹，避免题目调整后把前后指标直接混用。"""
    raw = path.read_bytes()
    data = json.loads(raw)
    revision = data.get("revision") if isinstance(data, dict) else None
    return {"dataset_revision": revision, "dataset_sha256": hashlib.sha256(raw).hexdigest()}


def load_cases(path: Path) -> list[dict]:
    """校验人工标注；正例允许多个合理来源，负例要求无静态证据。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("RAG 挑战集必须包含非空 cases 列表")
    known_sources = {doc.source for doc in load_approved_corpus()}
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or any(
            not isinstance(case.get(field), str) or not case[field].strip()
            for field in ("id", "group", "query")
        ):
            raise ValueError("RAG 用例必须包含非空 id、group、query")
        if case["id"] in ids:
            raise ValueError(f"RAG 用例 ID 重复：{case['id']}")
        ids.add(case["id"])
        sources = case.get("sources")
        if (not isinstance(sources, list) or any(
            not isinstance(source, str) or source not in known_sources for source in sources
        )):
            raise ValueError(f"RAG 用例 {case['id']} 的预期来源不在白名单知识库中")
    return cases


def _percentile(values: list[float], percent: float) -> float:
    """计算耗时样本的指定百分位数。"""
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(percent * len(ordered)) - 1)], 2)


def evaluate_cases(cases: list[dict], retriever: HybridRetriever, top_k: int = 4) -> dict:
    """只计检索器 search 耗时；模型加载和首次预热不计入延迟分位。"""
    if not cases or top_k <= 0:
        raise ValueError("RAG 评测需要非空用例和正整数 top_k")
    warmup_started = time.perf_counter()
    retriever.search(cases[0]["query"], top_k=top_k)
    warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 2)
    rows = []
    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case["query"], top_k=top_k)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        sources = [hit["source"] for hit in hits]
        expected = case["sources"]
        rank = next((index + 1 for index, source in enumerate(sources)
                     if source in expected), None) if expected else None
        passed = rank is not None if expected else not hits
        rows.append({
            "id": case["id"], "group": case["group"], "query": case["query"],
            "expected_sources": expected, "observed_sources": sources,
            "top_doc_id": hits[0]["doc_id"] if hits else None,
            "top_score": hits[0]["score"] if hits else None,
            "top_rerank_score": hits[0].get("rerank_score") if hits else None,
            "expected_rank": rank, "passed": passed, "latency_ms": elapsed_ms,
        })
    positives = [row for row in rows if row["expected_sources"]]
    negatives = [row for row in rows if not row["expected_sources"]]
    groups = sorted({row["group"] for row in rows})
    return {
        "metrics": {
            "count": len(rows), "positive_count": len(positives),
            "negative_count": len(negatives),
            f"recall@{top_k}": round(sum(row["expected_rank"] is not None
                                       for row in positives) / len(positives), 4)
            if positives else None,
            "mrr": round(sum(1 / row["expected_rank"] for row in positives
                             if row["expected_rank"] is not None) / len(positives), 4)
            if positives else None,
            "no_evidence_rate": round(sum(not row["observed_sources"] for row in negatives)
                                      / len(negatives), 4) if negatives else None,
            "pass_by_group": {group: {
                "passed": sum(row["passed"] for row in rows if row["group"] == group),
                "count": sum(row["group"] == group for row in rows),
            } for group in groups},
            "query_latency_p50_ms": round(statistics.median(row["latency_ms"] for row in rows), 2),
            "query_latency_p95_ms": _percentile([row["latency_ms"] for row in rows], 0.95),
            "warmup_ms": warmup_ms,
        },
        "cases": rows,
    }


def compare(cases: list[dict], base: HybridRetriever, reranker, top_k: int = 4) -> dict:
    """共享一次构建的向量索引，避免开/关精排时语料或召回候选发生变化。"""
    without = HybridRetriever(
        base.documents, base.dense_search, min_score=base.min_score,
        dense_min_score=base.dense_min_score, backend=base.backend,
    )
    with_reranker = HybridRetriever(
        base.documents, base.dense_search, reranker=reranker, min_score=base.min_score,
        dense_min_score=base.dense_min_score, backend=base.backend,
    )
    baseline = evaluate_cases(cases, without, top_k=top_k)
    reranked = evaluate_cases(cases, with_reranker, top_k=top_k)
    return {
        "backend": base.backend, "top_k": top_k,
        "corpus_chunks": len(base.documents),
        "baseline": baseline, "reranked": reranked,
        "changed_top1": [before["id"] for before, after in zip(
            baseline["cases"], reranked["cases"]
        ) if before["top_doc_id"] != after["top_doc_id"]],
        "limitations": [
            "人工标注只核对来源或空命中，不验证片段是否足以完整回答，也不评估生成模型。",
            "部分负例与边界说明文档词面相关；有命中不等于生成模型必然编造，但不能从静态库直接回答实时/个人事实。",
            "延迟仅含预热后的单条 search；不含模型下载、索引建立、页面与生成模型延迟，受本机 CPU 和缓存影响。",
        ],
    }


def main() -> None:
    """比较不同检索配置的质量与耗时。"""
    parser = argparse.ArgumentParser(description="对比 FAQ RAG 开启/关闭 bge-reranker")
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).parents[2] / "evals" / "rag_challenge.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output", type=Path, help="详细 JSON 报告路径；默认写入 evals/reports")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    setup_started = time.perf_counter()
    base = default_retriever()
    if base.reranker is not None:
        parser.error("比较脚本须在 RAG_RERANKER=none 下运行，避免基线提前加载精排模型")
    index_setup_ms = round((time.perf_counter() - setup_started) * 1000, 2)
    reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_started = time.perf_counter()
    reranker = BGEReranker(reranker_model)
    reranker_setup_ms = round((time.perf_counter() - reranker_started) * 1000, 2)
    report = compare(cases, base, reranker, top_k=args.top_k)
    report["dataset"] = str(args.dataset.resolve())
    report.update(dataset_identity(args.dataset))
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["reranker_model"] = reranker_model
    report["setup_ms"] = {"retriever_and_index": index_setup_ms,
                          "reranker": reranker_setup_ms}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output or Path(__file__).parents[2] / "evals" / "reports" / f"rag_{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(output), "backend": report["backend"],
        "setup_ms": report["setup_ms"],
        "baseline": report["baseline"]["metrics"],
        "reranked": report["reranked"]["metrics"],
        "changed_top1": report["changed_top1"],
        "baseline_failed": [case["id"] for case in report["baseline"]["cases"]
                            if not case["passed"]],
        "reranked_failed": [case["id"] for case in report["reranked"]["cases"]
                            if not case["passed"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
