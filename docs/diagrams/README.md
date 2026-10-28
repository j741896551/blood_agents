# 智慧献血智能体项目图

打开 [smart-blood-donation.drawio](smart-blood-donation.drawio) 可在 diagrams.net / draw.io 中编辑三个页面：

1. **设计视图**：展示主 Agent 导航、预约与积分子 Agent 跨轮续接、流程终态回交主图。这是控制流设计的概览，不是尚待实现功能清单；OCR 原型暂缓，未画成当前业务子流程。
2. **当前代码架构**：展示 Gradio/FastAPI/CLI 入口、会话隔离与前置静态问答缓存、SQLite Checkpointer（纯内存库使用 MemorySaver）、主导航与五个编译领域子图、平台模型 API 主备、Mock 工具层和 FAQ 检索。积分与预约均可跨轮；记录和 FAQ 为单轮；OCR 子图仅保留原型、暂缓开发。
3. **当前预约多轮流程**：概括预约意图识别、参数收集、历史和时段查询、预览确认、Mock 接口提交及等待用户下一轮。此页侧重新建预约示意，不覆盖已实现的预约查询、取消已提交预约和签到全部分支；详见代码及项目清单。

图中节点的 `data-source` 对应绘图时的源码快照。第 2 页的 BGE + FAISS / Milvus Lite 与 bge-reranker 是**已实现、经本地专项验证的可选检索后端**；默认仍用轻量关键词检索，不表示远端 Milvus 或生产级 RAG 已验证。模型生成通过第三方平台的 OpenAI 兼容 API；真实跨供应商备用切换尚未验证。**图中的 MCP“仅礼品目录”是 2026-10-21 快照**：当前代码已扩展为礼品、积分、献血记录三个只读工具，经 stdio MCP 转发至独立 Mock HTTP 服务；其余业务查询/写入走 HTTP adapter，图纸本轮尚未重新绘制。OCR 和 LoRA 当前暂缓；vLLM/TGI 只是 `deploy/` 历史模板，非当前开发或验收范围。

本图保留原有三页和节点布局，只校正已变化的技术事实。第 1 页是设计视角，第 2、3 页是代码视角；简化连线不等于完整函数调用轨迹。源码入口和功能边界以 [README 目录说明](../../README.md#代码目录怎么读)、[中文代码解读](../CODE_WALKTHROUGH.md) 与 [项目状态](../../PROJECT.md) 为准。

三个 `*-preview.png` 是由 [build_diagrams.py](build_diagrams.py) 按相同节点/边定义生成的**离线示意预览**，不是 draw.io 原生渲染。更新后可重新生成并做结构校验：

```bash
.venv312/bin/python docs/diagrams/build_diagrams.py
.venv312/bin/python /Users/jiangnian/.codex/skills/drawio-skill/scripts/validate.py docs/diagrams/smart-blood-donation.drawio --score
```

若本机安装 draw.io CLI，可另外导出原生 PNG 做视觉复核；本次只完成离线预览与结构校验，未把它们当作原生渲染证据。
