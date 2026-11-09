# 安装与配置

> 本页对应 2026-11-09 的本地集成快照。GitHub `dev` 此次只发布文档与图册。直接运行该分支的代码时，以同分支配置模板和依赖声明为准。

配置模板是根目录 [`.env.example`](../.env.example)。应用通过 python-dotenv 读取 `.env`，已有进程环境变量优先；修改配置后重启相应进程。不要把真实密钥写入文档或提交Git。

## 环境与依赖

项目最低Python3.10，本机功能回归使用Python3.12。Gradio固定为5.50.0，Pydantic约束为2.12系列；以 [pyproject.toml](../pyproject.toml) 为准。

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[dev,api,ui,mcp]'
cp .env.example .env
```

需要BGE、FAISS/Milvus及PDF解析时再安装 `.[rag,rag_ingest]`。OCR、训练依赖不是当前启动要求。`langgraph-checkpoint-sqlite` 是旧文件兼容测试的开发依赖，运行时使用项目的SQLAlchemy saver。

## 平台模型 API

| 主服务 | 备用服务 | 含义 |
| --- | --- | --- |
| `OPENAI_API_KEY` | `BACKUP_OPENAI_API_KEY` | 对应平台密钥 |
| `OPENAI_BASE_URL` | `BACKUP_OPENAI_BASE_URL` | 对应平台的OpenAI兼容端点 |
| `OPENAI_MODEL` | `BACKUP_OPENAI_MODEL` | 平台公开的模型标识 |
| `LLM_TIMEOUT_SECONDS=40` | `BACKUP_LLM_TIMEOUT_SECONDS=20` | 每次模型尝试的等待上限 |
| `LLM_MAX_CONCURRENCY=1` | `BACKUP_LLM_MAX_CONCURRENCY=1` | 每进程在途请求上限 |

`LLM_TOTAL_TIMEOUT_SECONDS=60` 是一次网关调用的主备共用预算。一轮对话可能调用导航、理解、工具规划、证据核验和回复等多个阶段，所以不是整轮60秒SLA。达到预算的底层线程可能仍在运行，日志会记录迟到结果。

熔断参数为 `LLM_FAILURE_THRESHOLD=2`、`LLM_CIRCUIT_COOLDOWN_SECONDS=30`。主备独立配置；变量名称不要求使用某一家平台。是否可用、配额和实际延迟需要使用者自行在平台验证。本项目不部署vLLM/TGI生成模型，`deploy/` 是未实跑历史模板。

## 外部业务模式

| 模式 | 必要配置 | 额外服务 |
| --- | --- | --- |
| `mock_file` | `BLOOD_AGENT_MOCK_STATE`，可选 `BLOOD_AGENT_MOCK_SEED` | 无，独立 `demo_business/` 直接管理JSON |
| `http` | `BLOOD_AGENT_LEGACY_API_URL`，可选TOKEN/TIMEOUT | 业务HTTP服务 |
| `mcp` | `BLOOD_AGENT_MCP_URL`，可选TOKEN/TIMEOUT | MCP服务；默认Streamable HTTP，也支持stdio |

模式通过 `BLOOD_AGENT_BUSINESS_TRANSPORT` 唯一选择，默认 `http`。全部业务读写跟随同一模式；模型API、RAG和Agent数据库不受它控制。模式间不自动回退，也不自动复制历史业务数据。详细配置及MCP服务端的HTTP/文件后端见 [业务模式说明](BUSINESS_MODES.md)。

## RAG

| 配置 | 含义 |
| --- | --- |
| `VECTOR_BACKEND=memory` | 内存词面检索，无向量模型；可改 `faiss` / `milvus` |
| `EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5` | 本地编码器标识，首次加载可能下载权重 |
| `RAG_MILVUS_URI=./data/blood_knowledge.db` | Milvus Lite本地文件；也可填远端地址 |
| `MILVUS_COLLECTION=blood_knowledge` | Collection基础名称，实际索引带内容/模型指纹 |
| `RAG_DENSE_MIN_SCORE=0.55` | 向量候选绝对阈值，需按数据评测调优 |
| `RAG_RERANKER=none` | 默认不精排，设 `bge` 使用BGE重排器 |
| `RERANKER_MODEL=BAAI/bge-reranker-base` | 可选本地精排模型 |
| `BLOOD_AGENT_RAG_WORKSPACE=./.local/rag` | 上传、草稿和发布清单目录 |
| `BLOOD_AGENT_QUERY_MAX_STEPS=8` | FAQ每轮模型决策上限，可设1至12 |

Milvus Lite请使用 `RAG_MILVUS_URI`，不要把本地文件路径赋给pymilvus自身的 `MILVUS_URI`。客户端保活由 `RAG_MILVUS_KEEPALIVE_SECONDS=300` 控制，与模型超时无关。FAISS/Milvus与本地BGE是知识检索组件，不是本地生成式大模型服务。

空库不加载BGE或创建空Collection。管理页发布前的单题/策略测试使用临时向量，不写正式索引；通过人工确认后才构建并发布。详见 [知识库页面](RAG_MANAGEMENT_UI.md)。

## 存储、日志与页面

- `BLOOD_AGENT_DB=./blood_agent.db`：Agent会话、标题摘要、删除标记、偏好和画像；六类表经SQLAlchemy读写SQLite。
- `BLOOD_AGENT_CHECKPOINT_DB`：可选的独立checkpoint文件；未指定时根据Agent库文件名生成。保存流程快照，不等同脱敏对话表。
- `BLOOD_AGENT_CONVERSATION_RETENTION_DAYS=0`：对话表保留配置。设正数只清理对应轮次与空摘要，不自动清理业务库、checkpoint或JSONL日志。
- `BLOOD_AGENT_TRACE_MODE=full|summary|off`，`BLOOD_AGENT_TRACE_FILE`：追踪级别和文件；full可能含个人上下文，仅用于合成数据调试。
- `BLOOD_AGENT_RISK_MODE=passthrough`：当前暂停额外内容风控调用；关键事实、身份和写入条件检查仍保留。
- `GRADIO_SERVER_NAME=127.0.0.1`、`GRADIO_SERVER_PORT=7861`：页面地址。`BLOOD_AGENT_GRADIO_SEND_CONCURRENCY=1` 控制发送事件并发。

启动：`.venv312/bin/python -m blood_agents.web`。API：`.venv312/bin/python -m uvicorn blood_agents.api:app --host 127.0.0.1 --port 8000`。

## 调试入口

```bash
.venv312/bin/python -m blood_agents.log.trace_viewer --help
.venv312/bin/python -m blood_agents.evals.conversation_evaluation --help
.venv312/bin/python scripts/live_local_service_load_test.py --help
```

先用 `trace_id` 关联一轮模型请求、输出、纠正和工具回执，再用 `session_id` 追踪多轮。服务不可用不等于写入一定失败，应先查询业务结果。排障和主备行为见 [模型容灾](MODEL_FAILOVER.md)；评测与压测分开见 [Excel评测](CONVERSATION_EVAL.md) 和 [吞吐计划](THROUGHPUT_PLAN.md)。
