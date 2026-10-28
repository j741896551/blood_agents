"""按会话或单轮查看本地 JSONL 追踪；不会调用模型或修改业务数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_FIELDS = (
    "user_question", "model_input", "model_output", "structured_result",
    "before_correction", "literal_slots", "after_correction", "answer",
)
SUMMARY_FIELDS = (
    "stage_zh", "stage", "provider", "status", "mode", "intent", "node",
    "tool", "error_type", "result_status", "result_code", "corrected_slots",
)


def read_events(path: Path) -> list[dict]:
    """跳过受损/非对象行；本地追踪日志可直接逐行追加。"""
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("trace_id"):
                events.append(event)
    return events


def format_events(events: list[dict], *, raw: bool = False) -> str:
    """保留英文键名以便对照代码，同时给每个节点显示中文事件名。"""
    rows = []
    for event in events:
        title = event.get("event_zh") or event.get("event") or "未知事件"
        rows.append(
            f"{event.get('ts', '?')}  {title} [{event.get('event', '?')}] "
            f"用户={event.get('user_id', '?')} 会话={event.get('session_id', '?')} "
            f"轮次={event.get('trace_id', '?')}"
        )
        details = {key: event[key] for key in SUMMARY_FIELDS if key in event}
        if details:
            rows.append("  摘要: " + json.dumps(details, ensure_ascii=False))
        if raw:
            # 原文只在 --raw 下展示；summary 模式下没有这些字段。
            for key in RAW_FIELDS:
                if key in event:
                    value = json.dumps(event[key], ensure_ascii=False, indent=2)
                    rows.append(f"  {key}: {value}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    """读取追踪日志并展示会话执行链路。"""
    parser = argparse.ArgumentParser(description="查看智慧献血智能体的中文执行追踪")
    parser.add_argument("--file", type=Path, default=Path("./logs/blood-agent-trace.jsonl"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session-id", help="查看一个会话的所有轮次")
    group.add_argument("--trace-id", help="查看指定单轮")
    group.add_argument("--latest", action="store_true", help="查看最后一轮（默认）")
    parser.add_argument("--raw", action="store_true", help="展示用户原话、模型输入输出与规则纠正值")
    args = parser.parse_args(argv)
    if not args.file.is_file():
        parser.exit(2, f"日志文件不存在：{args.file}。请先配置 BLOOD_AGENT_TRACE_FILE 并重启服务。\n")
    events = read_events(args.file)
    if not events:
        parser.exit(2, "日志文件中尚无可用的对话追踪。\n")
    if args.session_id:
        selected = [event for event in events if event.get("session_id") == args.session_id]
    else:
        # 超时的后台模型可能在下一轮开始后才返回；“最近一轮”按开始事件判定。
        latest_started = next(
            (event["trace_id"] for event in reversed(events) if event.get("event") == "turn.start"),
            events[-1]["trace_id"],
        )
        trace_id = args.trace_id or latest_started
        selected = [event for event in events if event.get("trace_id") == trace_id]
    if not selected:
        parser.exit(2, "没有找到对应的会话或轮次。\n")
    print(format_events(selected, raw=args.raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
