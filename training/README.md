# 意图识别数据准备（任务 7；LoRA 微调暂缓）

用户已决定跳过当前阶段的 LoRA 微调训练，暂不考虑微调。这里只保留此前完成的合成数据、划分与规则基线；不再把训练脚本、模型下载、权重或主图接入作为当前待办。

当前主导航的合法意图来自 `AgentRegistry.intents()`，包含 `general` 和已注册的五个领域。`ocr` 仍在注册表中，但真实图片识别已按项目决定暂缓；训练数据出现 `ocr` 标签不代表 OCR 功能已恢复。

`intent_dataset.jsonl` 是此前准备的**纯合成单轮分类样本**，每行只有 `text` 和 `intent`。若未来经用户确认恢复分类模型训练，其输出标签仍须由服务端适配为现有导航的 `{"intent":"..."}` 结构并经过注册表校验；当前代码没有接入 LoRA 模型。旧的五条 `intent_lora_samples.jsonl` 仅为早期格式示意，缺少 `general`，不能直接拿来报告训练或测试成绩。

无需模型即可复现数据划分与现有规则基线：

```bash
.venv312/bin/python training/intent_data.py split \
  --input training/intent_dataset.jsonl \
  --output-dir /private/tmp/blood-intent-split --seed 42
.venv312/bin/python training/intent_data.py rules \
  --input /private/tmp/blood-intent-split/test.jsonl \
  --report /private/tmp/blood-intent-rules.json
```

划分按类别固定随机种子，拒绝未知标签、重复话术、缺失类别和不足五条的类别；输出文件权限为 `0600`，且不会覆盖已有结果。当前 48 条样本划为训练 24、验证 12、测试 12；一次本地规则基线为 11/12，误判样本是“上一次实际献血是什么时候”被规则归为 FAQ。**这是合成小样本的规则路由结果，不是大模型或 LoRA 准确率，也不能用于宣称提升。**

当前不推进 PEFT `SEQ_CLS` LoRA 训练脚本或候选模型评测。若未来重新启动微调，需先确认数据脱敏与人工复核，防止训练/验证/测试话术泄漏，再单独记录模型下载、硬件、训练耗时和评测结果。生产或简历指标需在固定独立测试集上对比，不能用训练集成绩代替。
