"""主导航意图数据的本地划分与规则基线；不调用模型，也不训练权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from blood_agents.orchestration.agent_registry import default_agent_registry


def load_examples(path: Path, allowed: set[str] | None = None) -> list[dict[str, str]]:
    """按当前 Agent 注册表验证标签，拒绝重复话术和旧格式的裸字符串标签。"""
    allowed = allowed or set(default_agent_registry().intents())
    examples = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(row, dict) or set(row) != {"text", "intent"}:
            raise ValueError(f"第 {line_number} 行必须且只能包含 text、intent")
        text, intent = row["text"], row["intent"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {line_number} 行缺少有效话术")
        if not isinstance(intent, str) or intent not in allowed:
            raise ValueError(f"第 {line_number} 行意图未注册: {intent!r}")
        normalized = "".join(text.casefold().split())
        if normalized in seen:
            raise ValueError(f"第 {line_number} 行话术重复: {text!r}")
        seen.add(normalized)
        examples.append({"text": text.strip(), "intent": intent})
    if not examples:
        raise ValueError("意图数据集不能为空")
    return examples


def split_examples(examples: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """按类别固定随机种子划分；每类至少保留一条验证/测试样本。"""
    by_intent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in examples:
        by_intent[row["intent"]].append(row)
    result: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for intent in sorted(by_intent):
        rows = by_intent[intent].copy()
        if len(rows) < 5:
            raise ValueError(f"{intent} 只有 {len(rows)} 条；每类至少 5 条才能划分")
        rng.shuffle(rows)
        holdout = max(1, round(len(rows) * 0.25))
        # 极小类别仍保留训练样本，避免出现只有验证/测试、无法训练的类别。
        holdout = min(holdout, (len(rows) - 1) // 2)
        result["validation"].extend(rows[:holdout])
        result["test"].extend(rows[holdout:2 * holdout])
        result["train"].extend(rows[2 * holdout:])
    for rows in result.values():
        rng.shuffle(rows)
    return result


def _write_private_json(path: Path, value: object) -> None:
    """报告可能使用真实语料；只允许当前用户读取，且不覆盖已有结果。"""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _write_private_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """以仅当前用户可读的权限写入 JSONL，且不覆盖已有文件。"""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(input_path: Path, output_dir: Path, seed: int = 42) -> dict:
    """校验合成意图数据并按固定随机种子划分数据集。"""
    examples = load_examples(input_path)
    expected_intents = set(default_agent_registry().intents())
    missing_intents = expected_intents - {row["intent"] for row in examples}
    if missing_intents:
        raise ValueError(f"数据集缺少已注册意图: {', '.join(sorted(missing_intents))}")
    splits = split_examples(examples, seed)
    targets = [output_dir / f"{name}.jsonl" for name in splits]
    targets.append(output_dir / "manifest.json")
    if any(path.exists() for path in targets):
        raise FileExistsError("输出目录已有划分文件；为避免覆盖请使用新目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed,
        "labels": list(default_agent_registry().intents()),
        "counts": {name: dict(sorted(Counter(row["intent"] for row in rows).items()))
                   for name, rows in splits.items()},
        "limitations": "合成小样本仅验证工程链路；不能据此报告生产意图准确率或 LoRA 提升。",
    }
    for name, rows in splits.items():
        _write_private_jsonl(output_dir / f"{name}.jsonl", rows)
    _write_private_json(output_dir / "manifest.json", manifest)
    return manifest


def evaluate_rules(input_path: Path) -> dict:
    """固定测试集上运行现有离线规则路由；不是外部 LLM 或 LoRA 成绩。"""
    examples = load_examples(input_path)
    router = default_agent_registry()
    labels = list(router.intents())
    rows = [{**row, "predicted": router.rule_route(row["text"])} for row in examples]
    by_label = {}
    for label in labels:
        tp = sum(row["intent"] == row["predicted"] == label for row in rows)
        actual = sum(row["intent"] == label for row in rows)
        predicted = sum(row["predicted"] == label for row in rows)
        denominator = actual + predicted
        by_label[label] = {"support": actual, "precision": round(tp / predicted, 4)
                           if predicted else 0.0,
                           "recall": round(tp / actual, 4) if actual else 0.0,
                           "f1": round(2 * tp / denominator, 4) if denominator else 0.0}
    return {
        "mode": "offline_rules_baseline",
        "count": len(rows),
        "correct": sum(row["intent"] == row["predicted"] for row in rows),
        "accuracy": round(sum(row["intent"] == row["predicted"] for row in rows)
                          / len(rows), 4),
        "macro_f1": round(sum(item["f1"] for item in by_label.values()) / len(labels), 4),
        "by_label": by_label,
        "errors": [row for row in rows if row["intent"] != row["predicted"]],
        "limitations": "仅为本地规则路由在固定合成话术上的结果；未评测外部大模型或 LoRA。",
    }


def main() -> None:
    """解析命令行参数，执行数据划分或离线规则基线评测。"""
    parser = argparse.ArgumentParser(description="主导航意图数据划分与离线规则基线")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--input", type=Path, default=Path("training/intent_dataset.jsonl"))
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    baseline = commands.add_parser("rules")
    baseline.add_argument("--input", type=Path, required=True)
    baseline.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "split":
        result = prepare(args.input, args.output_dir, args.seed)
    else:
        result = evaluate_rules(args.input)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            _write_private_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
