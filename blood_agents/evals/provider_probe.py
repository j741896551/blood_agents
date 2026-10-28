"""对平台模型做有界、只读的连接与并发探测；默认仅打印计划，不发请求。"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from langchain_core.messages import HumanMessage, SystemMessage

from blood_agents.llm.model_gateway import failure_kind
from blood_agents.orchestration.router import IntentRouter

PROBE_PROMPT = "这是合成的 API 连通性测试。请只回复：OK"
MAX_OUTPUT_TOKENS = 128


def _percentile(values: list[float], fraction: float) -> float | None:
    """小样本使用最近秩，报告仅供本次环境对照，不推算供应商 SLA。"""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 2)


def run_probe(model: object, *, provider: str, parallel: int) -> dict:
    """只发送 parallel 条固定合成问题；不接触业务图，也不执行写工具。"""
    if parallel not in (1, 2, 4):
        raise ValueError("parallel must be 1, 2, or 4")
    gate = Barrier(parallel)

    def one(index: int) -> dict:
        """执行单次供应商探测并收集结果。"""
        started = time.monotonic()
        try:
            gate.wait(timeout=5)
            reply = model.invoke([
                SystemMessage(content="你在执行只读连通性测试。不要调用工具。"),
                HumanMessage(content=f"{PROBE_PROMPT} 测试序号 {index + 1}。"),
            ])
            content = reply.content if isinstance(reply.content, str) else ""
            usage = getattr(reply, "usage_metadata", None) or {}
            return {
                "status": "success" if content.strip() else "empty_response",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "content_chars": len(content),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
        except Exception as exc:  # noqa: BLE001 - 报告供应商/网络故障，但不输出异常原文
            error_type, _ = failure_kind(exc)
            status_code = getattr(exc, "status_code", None)
            return {
                "status": "failed",
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
                "error_type": error_type,
                "exception_class": type(exc).__name__,
                "http_status": status_code if isinstance(status_code, int) else None,
            }

    wall_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        results = list(executor.map(one, range(parallel)))
    durations = [item["duration_ms"] for item in results]
    return {
        "provider": provider,
        "requests": parallel,
        "success": sum(item["status"] == "success" for item in results),
        "wall_ms": round((time.monotonic() - wall_started) * 1000, 2),
        "p50_ms": _percentile(durations, 0.5),
        "p95_ms": _percentile(durations, 0.95),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    """运行模型服务的显式连通性探测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("primary", "backup"), required=True)
    parser.add_argument("--parallel", type=int, choices=(1, 2, 4), default=1,
                        help="单次最多 4 个真实 API 请求；阶梯测试需逐级运行")
    parser.add_argument("--execute", action="store_true", help="显式允许发送固定合成问题")
    args = parser.parse_args(argv)

    router = IntentRouter.from_env()
    configured = {item.name: item for item in getattr(router.llm, "providers", [])}
    selected = configured.get(args.provider)
    if selected is None:
        parser.error(f"{args.provider} is not configured")

    plan = {
        "provider": args.provider,
        "model": selected.model.model_name,
        "requests": args.parallel,
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "timeout_seconds_per_request": selected.timeout,
        "sends_real_api_requests": args.execute,
        "business_tools": "disabled",
    }
    if not args.execute:
        print(json.dumps({"status": "dry_run", "plan": plan}, ensure_ascii=False))
        return 0

    # 使用现有平台客户端配置，但限制输出；探测的是供应商容量，不受应用默认在途=1 干扰。
    model = selected.model.bind(max_tokens=MAX_OUTPUT_TOKENS)
    report = run_probe(model, provider=args.provider, parallel=args.parallel)
    print(json.dumps({"plan": plan, "report": report}, ensure_ascii=False))
    return 0 if report["success"] == args.parallel else 1


if __name__ == "__main__":
    raise SystemExit(main())
