# 血站智慧献血导航 · 作品集证据矩阵

核对日期：2026-10-23。本文是作品集讲解用的**证据索引**，不是第二份任务清单；任务状态和范围以 [`PROJECT.md`](../PROJECT.md) 为准。所有用户、预约、积分、订单和献血记录均为本地合成 Mock 数据，不连接公司旧业务接口。简历原项目的准确率、召回率、自动化率或并发指标不视为本仓库实测值。

## 当前版本可展示什么

| 技术/能力 | 当前实现与定位 | 本地证据 | 仍不能宣称 |
|---|---|---|---|
| 多智能体路由与协作 | [`agent_registry.py`](../blood_agents/orchestration/agent_registry.py) 显式登记领域能力；[`graph.py`](../blood_agents/orchestration/graph.py) 将 Supervisor 接到预约、积分、记录、FAQ、OCR 五个编译子图。预约/积分可跨轮等待和回交。 | [`test_conversation_tools.py`](../tests/test_conversation_tools.py)、[`test_agent_registry.py`](../tests/test_agent_registry.py)、[`test_persistent_checkpoint.py`](../tests/test_persistent_checkpoint.py) 覆盖派发、交接、隔离与重建续接。 | 运行时自动扫描/热插拔子 Agent；大样本意图准确率。OCR 仅保留原型，当前暂缓。 |
| 模型理解与自然回复 | [`router.py`](../blood_agents/orchestration/router.py) 分离主导航与领域理解；[`reply_composer.py`](../blood_agents/llm/reply_composer.py) 依据结构化工具结果组织回复。无可用模型时业务答复明确不可用。 | 本地可控模型、HTTP Mock 与历史小样本真实主模型记录见 `PROJECT.md`；关键回执和无回执成功宣称有针对性测试。 | 所有自然语言幻觉均已消除；当前版本所有领域在真实模型页面多轮稳定通过。 |
| 预约、积分、记录 | [`domains/`](../blood_agents/domains/) 负责子 Agent 流程，[`legacy_api.py`](../blood_agents/business/legacy_api.py) 与 [`repository.py`](../blood_agents/business/repository.py) 提供 Mock 接口/持久数据。提交、取消、签到、兑换等写入须经预览确认，并以接口回执为准；预约确认另要求回复节点成功生成并校验预览，服务端保留可确认标记。 | [`test_appointment_lifecycle.py`](../tests/test_appointment_lifecycle.py)、[`test_points_extended.py`](../tests/test_points_extended.py)、[`test_records_agent.py`](../tests/test_records_agent.py) 及其他图级/API/Gradio 回调测试。当前版无模型页面复测“预览不可用→确认”无预约写入。 | 真实血站预约、支付、物流或个人献血记录；业务系统内部结算与库存规则的生产正确性。服务端标记不等于浏览器实际收件回执。 |
| RAG 与证据交接 | [`rag.py`](../blood_agents/knowledge/rag.py) 提供默认关键词及可选 BGE+FAISS/Milvus Lite；[`adapters.py`](../blood_agents/knowledge/adapters.py) 接编码与可选重排；[`evidence.py`](../blood_agents/knowledge/evidence.py) 核验候选证据后才交给 FAQ 回复。个人下次献血日期再结合本轮记录工具与间隔规则计算参考日期。 | 本地真实 BGE 编码/FAISS/Milvus Lite 入库和重排预测已有验证；历史 `rag_evidence_20261020T103827Z-a62ba481.json` 为 28 条合成挑战题，18/18 正例来源命中、10/10 负例拒答、故障 0；当前个人日期组合有可控图级测试。 | 29 题扩充版的真实模型成绩、医学事实完整性、最终答复的医学正确性、远端 Milvus 或生产知识库。28 条结果不是通用 RAG 准确率。 |
| MCP 工具协议 | [`mcp_server.py`](../blood_agents/business/mcp_server.py) / [`mcp_client.py`](../blood_agents/business/mcp_client.py) 将礼品、积分和献血记录三个只读查询经 MCP 协议转发至同一 Mock 业务 HTTP 服务；其余查询/写入由 HTTP adapter 承担。 | [`test_mcp_protocol.py`](../tests/test_mcp_protocol.py) 的进程内协议与图级回归；隔离临时库的真实本机双进程烟测验证 U1001/U1002 的回执、`mcp_stdio` 轨迹及停服后的结构化错误。 | 22 个注册工具都已 MCP 化；网络 MCP Server、真实模型页面、生产吞吐或鉴权。 |
| 平台 API 容灾与并发 | [`model_gateway.py`](../blood_agents/llm/model_gateway.py) 封装主备调用、超时预算、熔断和进程内在途上限；[`conversation.py`](../blood_agents/orchestration/conversation.py) 同会话串行，跨会话可并行。 | [`test_model_gateway.py`](../tests/test_model_gateway.py) 的故障注入与 HTTP Mock 切换、FastAPI 并发回归；2026-10-23 增补悬挂主请求、备用成功/失败、迟到结果和流式首片前后超时测试。2026-10-21 固定合成只读主平台直连 1/2/4 路共 7/7 请求成功（本地忽略的 `provider_probe_20261021.json`）。 | 真实备用模型接续成功；当前 `/chat` 的 4 路吞吐、平台额度/SLA、多 Worker 一致性。该探测绕过业务图和应用在途上限；备用北京端点超时，国际端点对当前 Key 返回 401。 |
| 保守流式回复 | [`model_gateway.py`](../blood_agents/llm/model_gateway.py)、[`reply_composer.py`](../blood_agents/llm/reply_composer.py) 与 [`web.py`](../blood_agents/web.py) 只让主导航的普通 `general` 回复逐片显示；问候能力清单、业务、FAQ 与需确认的回复仍完整校验后显示。 | [`test_model_gateway.py`](../tests/test_model_gateway.py)、[`test_reply_composer.py`](../tests/test_reply_composer.py)、[`test_web_history_restore.py`](../tests/test_web_history_restore.py) 覆盖首片前后故障、意图隔离及草稿覆盖。2026-10-23 隔离假回复 Gradio 页面逐点击看到导航先出第一段，预约/FAQ 等待后整段出现。 | 真实平台的首片延迟、断线取消或业务回复流式。已显示片段无法从用户视野中撤回；隔离页面测试没有运行真实 LangGraph/模型供应商链路。 |
| 会话、记忆与静态缓存 | [`conversation.py`](../blood_agents/orchestration/conversation.py) 与 SQLite checkpoint 保存会话状态；[`long_term_memory.py`](../blood_agents/orchestration/long_term_memory.py) 只保存经原话核验的显式偏好；[`static_faq_cache.py`](../blood_agents/knowledge/static_faq_cache.py) 对完整唯一命中的已确认演示说明跳过模型，纯缓存会话标题也不送模。 | [`test_persistent_checkpoint.py`](../tests/test_persistent_checkpoint.py)、[`test_long_term_memory.py`](../tests/test_long_term_memory.py)、[`test_static_faq_cache.py`](../tests/test_static_faq_cache.py) 覆盖跨会话、撤销、活动流程回退与零模型调用。 | 可信用户登录、完整个人数据删除、真实模型偏好提取质量、语义相似问答缓存或生产级隐私合规。 |
| 可观察与自动评测 | JSONL 中英双语 trace 关联 `user_id`/`session_id`/`trace_id`；[`conversation_evaluation.py`](../blood_agents/evals/conversation_evaluation.py) 固定多轮 Excel 话术，检查模型路径、节点/工具/回执并执行整场模型评审。预约预览另检查模型回复与服务端可确认状态。Gradio 有历史侧栏和执行摘要；流式 trace 记录首片耗时、片段数和输出字符数。 | 2026-10-23 Python 3.12 临时 Mock 库 `pytest -q`：**394 passed, 34 warnings**；`ruff check .` 通过。先前 `pip check` 无依赖冲突；无模型 Mock 页面验证欢迎、侧栏/隔离、缓存与 checkpoint 续接，隔离假回复页面验证流式显示；预约 Excel 负向离线诊断确认未交付预览后无写入。历史报告见下节。 | 大样本对话成功率、模型评审即人工事实审核、当前版真实模型页面多轮、真实流式指标和备用供应商验收。 |

## 有数字的证据如何讲

- **当前回归**：394 项 pytest 是单元/集成/可控模型及 Mock 覆盖，不是 394 条真实模型对话；34 条依赖/Gradio 弃用等警告需在后续升级时复核。Ruff 和先前的 `pip check` 是代码/依赖检查，不证明真实平台可用。无模型页面只证明本地编排与不可用边界；隔离假回复页面只证明 Gradio 分段显示，不证明真实模型标题、流速或预约成功链路。
- **积分历史模型批测**：2026-10-15 的 4 场景、21 轮，逐轮 19/21、场景评审 3/4，整体失败。报告 `evals/reports/20261015T053931Z-4787af40.json`/`.xlsx`；这是当时用例和版本的结果，后续定向修复与本地测试不能将该批测改写为全通过。项目中的这些报告默认被 Git 忽略，发布作品集前须单独筛选、脱敏与附带。
- **FAQ 历史模型批测**：2026-10-20 的修订用例 `20261020T080756Z-e9acd9b6.json` 为 6/6 轮、4/4 场景通过；后续新增个人间隔路由和静态缓存后未重新运行同一真实模型批测，不能直接代表当前版本。
- **RAG 28 题**：仅核对合成问题的来源命中/拒答和证据核验流程，不逐题验证最终医学回答。公开摘要和虚构演示文档均非实时法规接口。
- **平台探测 7/7**：只是固定合成只读提示的极小直连样本；备用路径未打通，不能报告“自动容灾实测成功”或“支持四并发用户”。

## 暂缓与下一项演示校对

OCR 真实识别、LoRA 微调和 vLLM/TGI 本地生成模型部署均不在当前交付范围；文件/模板存在不等于技术已复现。内容风控模型按用户要求暂停，默认透传；身份脱敏、工具回执与写操作确认仍保留。当前没有真实旧业务接口或可信登录认证。

[`DEMO_WALKTHROUGH.md`](DEMO_WALKTHROUGH.md) 已按当前 Python 3.12、缓存节点、欢迎/会话侧栏及流式 A 模式校对，并记录 2026-10-21 无模型 Mock 页面和 2026-10-23 隔离假回复流式页面的不同验证边界；预约 Excel 已补强正向预览断言，离线负向诊断不再可能被表述为预约成功。旧日期真实模型浏览器记录仍是历史证据。下一步由用户按演示说明在隔离合成 Mock 会话中试用当前版真实模型，记录实际预览、确认回执、标题/摘要及流式首片的 trace；真实平台密钥、费用与备用接续亦由用户自行验证，不把本报告当成生产验收。
