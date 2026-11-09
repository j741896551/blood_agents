"""LoRA 前的数据契约与离线基线；不下载模型或训练权重。"""

import json
import stat
from pathlib import Path

import pytest

from blood_agents.orchestration.agent_registry import default_agent_registry
from training.intent_data import evaluate_rules, load_examples, prepare, split_examples


def test_synthetic_dataset_matches_registered_intents_and_split_is_reproducible(tmp_path):
    source = Path("training/intent_dataset.jsonl")
    examples = load_examples(source)
    assert len(examples) == 48
    assert {row["intent"] for row in examples} == set(default_agent_registry().intents())
    first = split_examples(examples, seed=42)
    assert first == split_examples(examples, seed=42)
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 24, "validation": 12, "test": 12,
    }
    assert {row["text"] for rows in first.values() for row in rows} == {
        row["text"] for row in examples
    }
    assert all(len({row["text"] for row in rows}) == len(rows) for rows in first.values())
    for intent in default_agent_registry().intents():
        assert all(any(row["intent"] == intent for row in rows) for rows in first.values())

    output = tmp_path / "split"
    manifest = prepare(source, output, seed=42)
    assert manifest["seed"] == 42
    assert manifest["counts"]["test"] == {intent: 2 for intent in sorted(
        default_agent_registry().intents())}
    assert stat.S_IMODE((output / "manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "test.jsonl").stat().st_mode) == 0o600
    assert load_examples(output / "test.jsonl") == first["test"]
    with pytest.raises(FileExistsError, match="已有划分文件"):
        prepare(source, output, seed=42)


def test_dataset_rejects_unknown_intent_and_duplicate_text(tmp_path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps({"text": "你好", "intent": "invented"}) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="意图未注册"):
        load_examples(invalid)
    invalid.write_text("\n".join([
        json.dumps({"text": "你好", "intent": "general"}),
        json.dumps({"text": " 你 好 ", "intent": "general"}),
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="话术重复"):
        load_examples(invalid)


def test_split_rejects_too_few_examples_per_intent():
    with pytest.raises(ValueError, match="至少 5 条"):
        split_examples([{"text": "你好", "intent": "general"}], seed=42)


def test_prepare_requires_every_registered_intent(tmp_path):
    source = tmp_path / "only-general.jsonl"
    source.write_text("\n".join(json.dumps({"text": f"问候{i}", "intent": "general"})
                                for i in range(5)), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少已注册意图"):
        prepare(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_rules_baseline_reports_errors_without_claiming_model_accuracy(tmp_path):
    source = Path("training/intent_dataset.jsonl")
    prepare(source, tmp_path, seed=42)
    report = evaluate_rules(tmp_path / "test.jsonl")
    assert report["mode"] == "offline_rules_baseline"
    assert report["count"] == 12
    assert 0 <= report["accuracy"] <= 1
    assert 0 <= report["macro_f1"] <= 1
    assert sum(row["support"] for row in report["by_label"].values()) == 12
    assert report["correct"] + len(report["errors"]) == 12
    assert "未评测外部大模型或 LoRA" in report["limitations"]
