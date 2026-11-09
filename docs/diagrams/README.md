# 架构图与流程图

更新于 2026-11-09。本地图谱快照。GitHub dev 此次仅发布文档，部分功能仍在本地集成。第 12 页为待实现设计。

[可编辑 draw.io](smart-blood-donation.drawio) · [离线图册](index.html) · [系统说明](../ARCHITECTURE.md) · [构建验证](VALIDATION.md)

PNG/SVG 是同一份节点、文字和正交几何生成的离线预览，**不是 draw.io 原生导出**。Mermaid 为可维护文本源；没有依赖在线 CDN。

蓝色为编排，绿色为领域/工具，紫色为模型/存储，橙色为条件/外部服务，灰色虚线为暂缓或规划。实线关系有代码来源；图中聚合节点代表同一职责的组件集合。

| 页 | 内容 | 文件 |
| --- | --- | --- |
| 01 | 系统全景 | [PNG](01-system-context.png) / [SVG](01-system-context.svg) / [Mermaid](01-system-context.mmd) |
| 02 | 主导航与多智能体编排 | [PNG](02-orchestration.png) / [SVG](02-orchestration.svg) / [Mermaid](02-orchestration.mmd) |
| 03 | 预约全流程 | [PNG](03-appointment.png) / [SVG](03-appointment.svg) / [Mermaid](03-appointment.mmd) |
| 04 | 爱心积分与纪念品 | [PNG](04-points.png) / [SVG](04-points.svg) / [Mermaid](04-points.mmd) |
| 05 | 记录查询与组合只读问答 | [PNG](05-records-faq.png) / [SVG](05-records-faq.svg) / [Mermaid](05-records-faq.mmd) |
| 06 | RAG 知识库生命周期 | [PNG](06-rag-pipeline.png) / [SVG](06-rag-pipeline.svg) / [Mermaid](06-rag-pipeline.mmd) |
| 07 | 业务接口与三种后端模式 | [PNG](07-business-transports.png) / [SVG](07-business-transports.svg) / [Mermaid](07-business-transports.mmd) |
| 08 | 会话、记忆与缓存 | [PNG](08-memory-cache.png) / [SVG](08-memory-cache.svg) / [Mermaid](08-memory-cache.mmd) |
| 09 | 模型容灾、并发与回复边界 | [PNG](09-failover-concurrency.png) / [SVG](09-failover-concurrency.svg) / [Mermaid](09-failover-concurrency.mmd) |
| 10 | 工具注册目录 | [PNG](10-tool-catalog.png) / [SVG](10-tool-catalog.svg) / [Mermaid](10-tool-catalog.mmd) |
| 11 | 结构化状态与数据流 | [PNG](11-state-dataflow.png) / [SVG](11-state-dataflow.svg) / [Mermaid](11-state-dataflow.mmd) |
| 12 | 待实现：统一临时确认机制 | [PNG](12-planned-confirmation.png) / [SVG](12-planned-confirmation.svg) / [Mermaid](12-planned-confirmation.mmd) |

## 阅读与来源

### 系统全景

这是系统边界图。业务数据不属于 Agent 数据库；大模型由平台 API 提供，本地 BGE 只承担编码与精排。

本地快照源码位置 `blood_agents/orchestration/conversation.py:23`；`blood_agents/business/transport.py:8`。

### 主导航与多智能体编排

每轮都会经过主导航；活动业务通过 checkpoint 续接。领域子图不是并行 fan-out，主图也没有独立 Skill 加载器。 图中领域派发框聚合两种路径：单任务由supervisor直派，多任务才经TaskCoordinator；不是所有请求都执行队列节点。

本地快照源码位置 `blood_agents/orchestration/graph.py:618`；`blood_agents/orchestration/agent_registry.py:46`。

### 预约全流程

收集无固定字段顺序，任何修改会撤销旧提交预览；局部确认只选网点。取消、签到有各自的待确认目标和回执，不能与退出未提交意向混称。

本地快照源码位置 `blood_agents/domains/appointment.py:372`；`blood_agents/orchestration/graph.py:415`。

### 爱心积分与纪念品

地址保存确认与兑换下单确认是两个动作。当前只传接口已有的介绍、规格与图片，不把描述性型号猜成可下单 SKU。

本地快照源码位置 `blood_agents/domains/points.py:41`；`blood_agents/llm/points_query_reply.py:10`。

### 记录查询与组合只读问答

个人下次献血参考日期不走固定关键词工作流；模型自行决定查询记录和规则，再引用来源做运算。无充分证据不能生成医学资格结论。

本地快照源码位置 `blood_agents/llm/query_agent.py:92`；`blood_agents/domains/records.py:8`。

### RAG 知识库生命周期

memory 是内存词面检索，不是向量数据库，也不是长期记忆。当前融合不是 BM25 或 RRF；草稿不进入聊天，文档冲突没有自动版本优先级硬过滤。

本地快照源码位置 `blood_agents/knowledge/rag.py:83`；`blood_agents/knowledge/workspace.py:248`；`blood_agents/knowledge/workspace.py:423`。

### 业务接口与三种后端模式

HTTP 演示由独立进程监听 8001，和 Gradio 分开。MCP 服务端可选文件后端；切换模式不迁移历史地址或订单。local 仅为旧测试兼容。

本地快照源码位置 `blood_agents/business/transport.py:8`；`blood_agents/business/mcp_server.py:116`。

### 会话、记忆与缓存

标题摘要不是预约授权；缓存仅适用于无个人/实时业务依赖的标准问答。会话表与 checkpoint 为不同 SQLite 文件，保留期清理不会自动删除所有存储。

本地快照源码位置 `blood_agents/orchestration/conversation.py:216`；`blood_agents/storage/models.py:13`。

### 模型容灾、并发与回复边界

默认主40秒、备20秒、单次预算60秒，不是整轮对话 SLA。超时后台请求可能稍后结束；普通 general 的保守流式不代表业务回答支持分段风控。

本地快照源码位置 `blood_agents/llm/model_gateway.py:141`；`blood_agents/llm/reply_composer.py:1060`。

### 工具注册目录

这是组件目录，不表示这些工具并行执行。模型只读循环仅能看到显式 consultation 白名单，个人工具由服务端注入 user_id；写工具只能由领域确认流程调用。

本地快照源码位置 `blood_agents/business/tools.py:158`。

### 结构化状态与数据流

action=confirm 是模型动作，不等于预约已经成功；agent_status=completed 表示流程结束，也可能是退出或失败。当前仍混有固定词和正则校正。

本地快照源码位置 `blood_agents/core/models.py:108`；`blood_agents/orchestration/router.py:427`。

### 待实现：统一临时确认机制

这是设计视图；当前没有通用 ConfirmationItem 队列。普通字段不逐个确认，修改优先于同句中的确认，模型语义准确率仍需真实多轮评测。

本地快照源码位置 `PROJECT.md:55`。

### 一轮消息：主导航与各领域分支

alt代表不同领域分支，不是一个请求同时运行全部Agent。每轮结束后保存状态，后续输入再续接；OCR只保留原型注册。

[Mermaid时序源](13-runtime-sequence.mmd) · [内嵌预览](SEQUENCES.md)

### 跨会话并发与平台主备边界

并行发生在不同会话请求间；相同会话由同一进程内锁串行。示例表达可配置在途预算，不表示默认额度能同时容纳所有请求，也没有多Worker分布式锁。

[Mermaid时序源](14-concurrency-sequence.mmd) · [内嵌预览](SEQUENCES.md)
