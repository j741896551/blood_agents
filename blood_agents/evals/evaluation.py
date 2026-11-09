from __future__ import annotations

import json
from pathlib import Path

from blood_agents.knowledge.rag import HybridRetriever, default_retriever
from blood_agents.orchestration.router import IntentRouter


def evaluate_routing(cases: list[dict], router: IntentRouter) -> dict:
    """运行意图路由样本并计算离线指标。"""
    correct = sum(router.route(case["text"]) == case["intent"] for case in cases)
    return {"count": len(cases), "accuracy": correct / len(cases) if cases else 0.0}


def evaluate_retrieval(cases: list[dict], retriever: HybridRetriever, top_k: int = 4) -> dict:
    """运行知识检索样本并计算离线指标。"""
    hits, reciprocal_ranks = 0, []
    negative_count = 0
    no_evidence_count = 0
    for case in cases:
        results = retriever.search(case["query"], top_k=top_k)
        if case.get("source") is None:
            negative_count += 1
            no_evidence_count += not results
            continue
        sources = [result["source"] for result in results]
        if case["source"] in sources:
            rank = sources.index(case["source"]) + 1
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    count = len(reciprocal_ranks)
    return {
        "count": len(cases),
        "positive_count": count,
        "negative_count": negative_count,
        f"recall@{top_k}": hits / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "no_evidence_rate": no_evidence_count / negative_count if negative_count else None,
    }


def run_evaluation(dataset_path: str) -> dict:
    """运行配置的离线评测并汇总指标。"""
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    return {
        "routing": evaluate_routing(data["routing"], IntentRouter()),
        "retrieval": evaluate_retrieval(data["retrieval"], default_retriever()),
    }


def main() -> None:
    """运行基础路由与检索离线评测。"""
    path = Path(__file__).parents[2] / "evals" / "dataset.json"
    print(json.dumps(run_evaluation(str(path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
