# 血站智慧献血导航

基于简历项目内容构建的 LangChain + LangGraph 多智能体系统。Agent 负责自然语言理解、领域分派、工具编排、跨轮状态及回复；业务事实和操作结果由外部接口返回。Agent 包保留 HTTP 客户端及 MCP 协议服务，不包含旧业务数据库或业务服务端。`local_business/` 是同级、被 Git 忽略的本机替身服务，供没有公司接口时运行与测试。RAG 支持公开资料检索与可选 FAISS/Milvus Lite+BGE+精排；OCR 与 LoRA 微调暂缓。

> 本仓库是可运行的工程化复现。简历中的“RAG 准确率 85%+、复杂问题召回率提升 30%+、流程自动化率提升 70%+”属于原项目成果描述；仓库提供对应评测方法，但不把这些数字伪装成本机 Demo 的实测结果。

第一次阅读项目建议先看 [中文代码解读](docs/CODE_WALKTHROUGH.md)，它按照一条消息的完整调用链解释 LangGraph、多轮预约、接口回执和 RAG。旧文档中的集成式业务仓储路径以本页“代码目录”和 `PROJECT.md` 最新状态为准。

## 代码目录怎么读

```text
blood_agents/
├── orchestration/  # 主图 graph、导航 router、注册表、跨轮会话
├── domains/        # appointment、points、records、faq、ocr 五个领域执行节点
├── llm/            # 主备模型网关、回复表达、工具规划、可选风控
├── business/       # 业务接口契约、HTTP 客户端、ToolRegistry、MCP stdio Server/客户端
├── storage/        # Agent 会话、摘要、偏好和画像存储
├── knowledge/      # 轻量 RAG 与 bge/FAISS/Milvus 可选适配器
├── core/           # 跨领域状态与数据契约
├── log/            # 中英双语追踪、脱敏与日志查看器
├── ui/             # 页面执行摘要格式化
├── evals/          # 单轮与多轮自动评测器
└── __init__.py / api.py / web.py / cli.py  # 包初始化与对外入口
local_business/       # 本机业务替身服务及数据仓储；被 Git 忽略
```

建议从 `web.py` 或 `api.py` 进入，接着读 `orchestration/graph.py` 的主图和子图接线，再按意图进入 `domains/` 对应文件。领域执行节点只给出结构化事实；最终回复统一走 `llm/reply_composer.py`。旧的 `blood_agents.graph` 等库导入由顶层 `__init__.py` 集中映射，新代码请使用分层路径；评测器与日志查看器的 `python -m` 命令需使用下方新路径。目录调整没有改变业务流程，也不代表新能力已实现。

## 系统架构

```text
自助设备 / Web / API
        │
LangGraph Supervisor ── 意图识别（模型 Prompt + 规则降级；LoRA 暂缓）
        │
        ├─ 预约 Agent ── 多轮槽位收集 ── 预约/时段工具 ── 回复节点
        ├─ 积分 Agent ── 查询/推荐/兑换/订单与地址 ── 接口回执 ── 回复节点
        ├─ 记录 Agent ── 历史查询 ────── 业务工具 ── 回复节点
        ├─ OCR Agent ─── 原型保留，真实图片引擎/输入暂缓
        └─ FAQ Agent ─── Hybrid RAG ────────────────── 回复节点
                         ├─ 多策略 Chunk
                         ├─ bge-large-zh-v1.5
                         ├─ FAISS / Milvus
                         └─ bge-reranker + 来源引用
        │
本地 ToolRegistry（19 个工具）── HTTP / 可选 MCP stdio 三项只读查询
        │
外部业务服务（本机无公司接口时可启动忽略目录中的替身服务）
        │
生成模型：第三方平台 API（主备独立配置，由 ModelGateway 调用）
```

## 已实现功能

- **多智能体编排**：五类领域 Agent 由 `AgentRegistry` 显式注册；能力描述、导航意图枚举、离线规则和 LangGraph 接线从同一清单生成。问候及能力询问由主导航回答；模型应按主题用 Markdown 项目列表介绍当前可用能力，若首稿写成长段落会请求模型改写一次，改写失败仍保留已通过事实校验的首稿。
- **多轮对话**：Agent 会话仓储和 LangGraph checkpoint 均以 SQLAlchemy 模型访问 SQLite，checkpoint 仍存独立文件，并按入口 `user_id + session_id` 隔离。每轮完成后另把文本和追踪元数据保存到 `conversation_turns`；页面按用户 ID 回显最近会话。纯内存测试可使用 `MemorySaver`。这些表不是旧业务表。
- **分层记忆（演示版）**：短期任务快照从已校验的 LangGraph 状态提取；长期记忆保存献血点/上午下午偏好，以及从用户稳定兴趣陈述中增量提取的爱好画像。模型候选需通过原话证据、字段白名单、临时/否定表达和隐私过滤，SQLite 保存画像快照、来源轮次、提取版本及覆盖游标；用户按 `user_id` 跨会话读取，明确不再喜欢/清除时可撤销。姓名、性别、血型、证件、联系信息和业务记录不属于画像；偏好不代替实时业务查询，也不授权预约。Gradio 在答复交付后/打开历史时懒更新，HTTP API 在响应后后台更新。完整唯一命中的静态标准问答缓存已接入，范围见下文。
- **旧系统编排**：业务查询和写操作经 Agent 侧接口契约及 HTTP 客户端完成；三个只读查询可经 MCP stdio 服务转发到同一 HTTP 接口。本机替身服务位于忽略目录，未连接或验证真实公司的旧接口。
- **献血人确认后提交**：Agent 从外部接口读取可用时段和历史献血人，收集本次献血人资料后展示待确认预览；用户明确确认才调用提交接口，成功或失败以接口回执为准。模型不可用时不交付可确认预览。
- **已提交预约查询**：预约子 Agent 按入口 `user_id` 调用只读接口查询已提交预约，不把历史偏好误当作本次预约结果。
- **预约生命周期**：新建、查询、取消已提交预约与签到经外部业务接口完成；取消和签到先查唯一目标、再经用户确认，状态与失败原因由回执决定。合成历史数据只存在本机替身目录，不随 Agent 包提交。
- **献血记录查询**：记录子 Agent 仅按入口 `user_id` 调用接口；记录日期、地点、献血类型、献血量等均来自当轮回执。全血毫升数与单采治疗量分开统计，未知字段不补造；完整身份资料不送入回复模型。公开资料只用于字段设计，不充当个人记录。
- **积分与兑换订单**：积分子图查询余额、礼品、订单和地址，支持推荐、兑换、地址管理与取消订单；新兑换核对礼品、积分和收货信息后再确认提交。余额、库存及订单状态只以接口回执为准，Agent 不实现旧业务结算规则。
- **模型表达层**：各领域子图和主导航都有独立回复节点；业务节点只提供结构化事实，由 LLM 组织最终回复。预约候选时段可按网点横向排列。除下述唯一命中的静态标准问答缓存外，无模型或模型服务失败时只报服务不可用；提交成败、身份泄露与日期等关键边界仍由程序约束，但不强制逐字复述接口内部提示。完整手机号/证件号先脱敏再核验回复。业务结果不由程序拼接成用户答复，静态缓存仅回复预先审核的 OCR 范围说明。模型不决定预约或积分操作结果。
- **保守流式显示**：Gradio 只对主导航的普通 `general` 回复逐片更新；问候/能力列表、预约、积分、献血记录、FAQ、静态缓存及需确认的操作仍在完整生成和校验后显示。流式首片前模型故障可尝试备用；首片后不拼接另一供应商文本，最终失败会用服务不可用答复覆盖页面草稿。已显示内容无法真正撤回，关闭页面也不保证供应商请求即时取消，因此这不是业务回复流式或内容安全保证。已有单测和隔离假回复页面逐点击验证显示机制；真实模型首片延迟、断线及平台接续尚未验证。
- **数据依据**：业务 Agent 无结构化查询/执行结果时不请求模型编造答复；提交与取消等操作结论须以工具回执为准。献血记录、OCR 和 FAQ 已分别区分有数据、空结果与查询/识别失败；FAQ 无证据时不能给出确定性的献血建议，检索失败不能说成知识库没有答案。现有校验是针对性门禁，不保证识别一切自然语言幻觉；跨领域及更大样本对抗评测仍在计划中。
- **预约工具规划**：预约子图中的模型规划节点只可从四个只读工具中提出请求，程序以入口用户身份执行；缺少确认/可用性校验所需事实时，业务节点仍会补做必要查询。模型不能请求或直接执行提交工具。
- **回复内容风控 Agent（按需开启）**：默认 `BLOOD_AGENT_RISK_MODE=passthrough`，不增加模型调用；设为 `model` 后，独立模型审查候选回复中的违法有害、辱骂、歧视、威胁和危险个体化医疗建议，否决后最多重写一次；最终拒答也须再经内容审查。内容风控不逐字审计业务字段；隐私与无回执成功宣称始终由确定性边界处理。仅因非关键字段/措辞遗漏产生的质量提示记入 `response_info.quality_warnings` 和日志，不截断回复。模型审查不保证法律合规；开启后会增加调用、延迟与费用，当前仅有可控模型测试，未完成真实供应商效果评测。
- **OCR（暂缓）**：保留现有 PaddleOCR/Tesseract 按需加载原型和子 Agent 四态处理，不删除已完成代码。用户当前没有图片识别场景，因此图片输入、真实引擎与后续编排不属于本阶段验收；现有路径仅接收服务端可访问的图片，网页直接上传和真实图片端到端均未验证。
- **知识问答 RAG（公开资料）**：默认语料为 11 篇公开资料人工摘要（10 个来源），不再包含虚构业务流程。FAQ 子 Agent 检索并核验片段证据后让模型回答；无证据或查询失败时不得用模型常识补造事实。`VECTOR_BACKEND=faiss|milvus` 可启用 BGE 向量检索，`RAG_RERANKER=bge` 可启用精排；默认仍为轻量词面检索。个人下次献血日期只在公开间隔证据与当轮个人记录均齐备时计算参考日期。旧版含虚构流程的评测报告不适用于当前语料。

要编辑默认知识库，请修改 `blood_agents/knowledge/documents/industry/` 的公开摘要，并在 `blood_agents/knowledge/corpus.json` 登记可核验的来源、日期和路径。个人业务数据与本地替身流程不应进入知识库；下载与入库过程见 [RAG 入库实操](docs/RAG_INGEST_WALKTHROUGH.md)。
- **工具生态**：当前统一注册 19 个工具。默认业务请求经 HTTP 客户端访问外部服务；`BLOOD_AGENT_BUSINESS_TRANSPORT=mcp` 时，礼品目录、积分与献血记录三个只读查询经 Agent 包内的 MCP stdio 服务转发到 HTTP 服务，其余查询与确认后的写操作仍经 HTTP。MCP 不暴露写操作。
- **模型平台 API**：`router.py` 根据主备 API Key、Base URL、模型名建立 `ChatOpenAI` 客户端，由 `ModelGateway` 为理解、工具规划、证据核验和回复提供故障切换。`deploy/` 的 vLLM/TGI 是未实跑的旧模板，不属于当前交付；Milvus Lite 属于 RAG 数据检索，不是生成模型部署。
- **自动评测**：原有路由 Accuracy、检索 Recall@K/MRR；另有逐轮运行 LangGraph 的多轮对话用例，校验意图、状态、工具、写操作确认和回执，并保存脱敏报告。离线模式只验编排/Mock，真实模型模式单独验模型理解与回复。

## 快速开始

模型理解分两层：Supervisor 用由 Agent 注册表生成的 `NavigationDecision` 枚举只选择领域；预约子图的 `appointment_understand` 用 `AppointmentUnderstanding` 提取预约参数和动作。两层分别校验 JSON 并可独立规则降级；完整的 `json` 代码块会先剥离围栏再校验，额外解释文字仍视为非法输出。注册是启动时显式进行的，不会扫描业务代码，也不支持运行中热插拔。
主导航和预约理解节点的系统提示词位于 `blood_agents/orchestration/router.py`；各领域回复提示词集中在 `blood_agents/llm/reply_composer.py`。执行 Agent 仍以确定性代码查询/提交，回复节点才使用模型表达；这是程序内集中管理，尚未外置为可编辑配置文件。详见 [代码解读](docs/CODE_WALKTHROUGH.md)。
用户修改预约参数后须重新确认。网页“本轮编排信息”分别展示理解模式 `model_info` 与回复模式 `response_info`；没有 API Key 时，非静态缓存轮次的回复模式为 `unavailable`，聊天只显示服务不可用错误，结构化状态仍用于排查和后续查询。

启用真实模型时，在 `.env` 中填写同一组主服务配置：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`，然后重启服务。变量名沿用 OpenAI 兼容接口约定，也可接 DeepSeek 等兼容服务。已有 `.env` 时直接编辑，不要用示例文件覆盖现有密钥。
例如使用 DeepSeek：`OPENAI_API_KEY` 填 DeepSeek Key，`OPENAI_BASE_URL=https://api.deepseek.com`，`OPENAI_MODEL` 填所用平台上实际可用的模型名。开发阶段模型等待配置为主服务最多 40 秒、备用最多 20 秒、每次网关调用整体最多约 60 秒；这是单个模型阶段的预算，不是整轮对话的上限，交付前需压测后调短。SDK 自动重试已关闭。`LLM_MAX_CONCURRENCY` 与 `BACKUP_LLM_MAX_CONCURRENCY` 分别限制本进程主备模型的在途请求数，默认 1；只有核实供应商额度并做小规模并发实测后再增大。达到上限会转备用或报服务不可用，并不表示模型平台必然不支持并发。
主备模型 Key 均未配置时仍可离线运行编排与工具回归；唯一命中的静态标准问答仍可直接回复 OCR 范围说明，但其他聊天不会产生业务答复，交互演示须配置可用模型。尚无可代表实际使用质量的真实模型准确率评测；已有小样本结果与局限见 `PROJECT.md`。
备用服务填写 `BACKUP_OPENAI_API_KEY`、`BACKUP_OPENAI_MODEL`、`BACKUP_OPENAI_BASE_URL`。
超时、连接失败、429/5xx 会切换备用服务，持续失败会熔断；理解阶段全部不可用时回退规则，回复阶段只返回服务不可用错误。若发生在提交之后，接口结果保存在本轮状态中，不重放写操作；用户应先查询结果再决定是否重试。
`model_info` 展示理解服务及尝试；`tool_plan_info` 展示预约只读工具请求；`response_info` 单独展示最终回复、风控放行/拒答或不可用。配置与容灾边界见 [模型容灾解读](docs/MODEL_FAILOVER.md)。
已完成本地故障注入验证，尚未完成真实多供应商联调；修改配置后须重启服务。
真实平台的只读连接/并发验证可参考 [平台 API 探测说明](docs/PROVIDER_PROBE.md)：命令默认不发请求，须在核对账号额度与费用后显式加 `--execute`。该探测不代替完整 `/chat` 主备切换验收。

完整 `/chat` 的 30/50 虚拟用户多轮混合负载、指标与扩容顺序见 [吞吐与并发验证计划](docs/THROUGHPUT_PLAN.md)。目前尚未运行该压测，不能将平台直连 1/2/4 路小样本当作应用吞吐结论。

项目最低版本已升级为 Python 3.10；当前使用 Python 3.12 验证。旧 `.venv` 若仍是 Python 3.9，不要用于**当前项目启动**：持久化 LangGraph 状态所需的 `langgraph-checkpoint-sqlite` 也安装在新版环境。首次使用时可新建独立环境：

作品集讲解时，先看 [实现与验证证据矩阵](docs/PORTFOLIO_EVIDENCE.md)；它把当前本地回归、历史真实模型小样本和未验证项分开，不以 Mock 结果代替公司系统或生产指标。

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[dev,api,ui,mcp]'
cp .env.example .env
.venv312/bin/python -m blood_agents.cli
```

启动 Gradio 聊天页面：

```bash
.venv312/bin/python -m blood_agents.web
```

如需演示“Agent 通过 HTTP 调用独立旧业务服务”，需启动两个本地进程（不是公司真实接口）：

1. 在 `.env` 中将 `BLOOD_AGENT_BUSINESS_TRANSPORT=http`、`BLOOD_AGENT_LEGACY_API_URL=http://127.0.0.1:8001`。
2. 终端一先启动本机业务替身：`.venv312/bin/python -m local_business.http_service`。默认监听 `127.0.0.1:8001`，数据保存在 `BLOOD_AGENT_LOCAL_BUSINESS_DB` 指定的独立文件（默认 `./local_business.db`）。这个目录不随 Agent 提交。
3. 终端二启动 Agent：`.venv312/bin/python -m blood_agents.web`（或照常运行 FastAPI API）。Agent 的业务工具通过 HTTP adapter 调用服务，工具轨迹标记 `legacy_http`。

未指定 transport 时默认走 HTTP，必须有可达的外部业务服务；本地替身只是开发选择。若要使用 MCP，将 `BLOOD_AGENT_BUSINESS_TRANSPORT=mcp`：积分、礼品和献血记录查询走 `Agent → MCP stdio 子进程 → 业务 HTTP 服务`，其他查询/写操作走 `Agent → 业务 HTTP 服务`，`tool_trace.backend` 对前三种查询标记为 `mcp_stdio`。MCP 子进程由 Agent 自动拉起。`BLOOD_AGENT_BUSINESS_TRANSPORT=local` 仅用于保留本机插件的测试，不是发布默认模式。Agent 的 `BLOOD_AGENT_DB` 只保存会话与偏好；本机替身使用独立的 `BLOOD_AGENT_LOCAL_BUSINESS_DB`。本机替身服务默认无认证且只绑定回环地址，不可暴露到公网。

已有旧版 `blood_agent.db` 不会被删除，也不会自动迁移其中的预约、订单等业务表。新替身默认使用新的 `local_business.db`，所以旧版业务记录不会自动出现在新服务中；若本机要临时沿用旧记录，可明确设置 `BLOOD_AGENT_LOCAL_BUSINESS_DB=./blood_agent.db`，但这只是过渡的同文件配置，不代表 Agent 又直接读写业务表。正式拆分数据应另做有备份、可核对的迁移。

当前运行时不再导入 `langgraph.checkpoint.sqlite`；该包只在开发测试中用于核验旧 checkpoint 文件兼容性。本机旧 `.venv` 为 Python 3.9，不能用于当前项目；请使用同一 `.venv312/bin/python` 安装依赖并启动。已有 `.env` 不必重复复制或覆盖。

本地 MCP 演示：按上段设置 `BLOOD_AGENT_BUSINESS_TRANSPORT=mcp` 并先启动本机替身 HTTP 服务，再询问“查询我的积分”或“查询我的献血记录”。礼品目录、积分和献血记录分别由 `blood_list_rewards`、`blood_get_points`、`blood_get_donation_records` 三个只读工具完成 `tools/list`/`tools/call`，其回执都来自 HTTP 服务。用户 ID 只由 Agent 入口上下文传给相应查询工具，不由模型任意指定；MCP 不暴露兑换、地址或预约写操作。当前每次调用会创建一个 stdio 子进程，适合协议演示，不代表高并发性能。2026-10-23 已用隔离临时库完成真实本机双进程无模型烟测：两个用户的积分/记录及 `mcp_stdio` 轨迹正确，停止业务服务后返回查询失败状态而不伪造空数据。无模型时只能验证节点和工具回执，自然业务聊天仍会回复服务不可用；真实模型页面交互由用户自行试用。

可复测双进程 HTTP/MCP 路径：运行 `.venv312/bin/python scripts/smoke_business_services.py`。脚本只绑定本机临时回环端口、使用临时合成数据库和本地预览测试模型，覆盖积分查询、确认后预约写入、跨用户隔离和 MCP stdio 献血记录查询；不会调用外部模型平台。运行环境需要允许绑定和访问 `127.0.0.1`，否则会明确报错并输出本机业务服务日志尾部。该烟测不能代替真实业务接口、模型质量或高并发验证。

浏览器访问启动时显示的地址（示例 `.env` 使用 `http://127.0.0.1:7861`）。左侧用户 ID 默认 `U1001`，可自行填写；页面打开或切换用户时，从 `conversation_turns` 读取该 ID 最近 30 个已保存会话，默认选中最近一条并回显其最近 100 轮。没有记录时创建一个未保存的新会话。通常首轮 AI 答复先显示，随后额外调用一次已配置的主/备模型，根据这组已脱敏问答生成短标题与摘要，写入 `conversation_session_summaries`，侧栏再更新为标题、轮次数与会话 ID 后缀；标题不再截取首句。**纯静态缓存会话例外**：侧栏使用固定的标准问答标题与摘要，不额外调用模型；以后该会话出现非缓存轮次才重新生成模型摘要。无模型、摘要模型失败或写入失败时显示“待生成标题”，聊天回复不受影响。已有旧会话在被打开时按需补生成一次，不会在页面启动时批量请求模型。这个标题与摘要只是页面元数据，尚不进入 Agent 提示词充当记忆。点击旧会话先只读加载历史，下一条新消息才按该用户+会话的 checkpoint 继续。点击“新对话”或 Chatbot 自带的清空图标会生成新会话 ID，**不会删除旧记录**；新会话发送第一条消息并完成保存后才成为持久列表项。手动输入会话 ID 仍保留在左侧折叠的调试区。页面同时展示示例问题和本轮编排信息，`tool_trace` 可查看工具名、脱敏业务参数、结果状态与耗时。

预约新建/取消/签到，以及积分兑换、取消兑换订单、保存/修改/设为默认收货地址、填写订单收货信息，在模型生成且服务端通过校验的待确认预览后，**对话卡片内、消息记录紧下方、输入框上方**会显示操作说明、对应的确认按钮与“放弃本次操作”按钮；无需滚动到输入框下方寻找。积分待确认回复会提示可点击按钮，也保留输入确认话术的方式。按钮先于侧栏标题摘要生成显示，标题模型响应慢时也不应拖延确认控件。点击按钮与输入原有确认/放弃话术走同一个 LangGraph 流程，不由前端直接写库；切换用户或会话、修改待确认信息、重复点击使旧按钮版本失效。新兑换若先保存收货地址，保存按钮只执行地址写入，随后会展示新的完整兑换预览，还需再次确认兑换。“放弃本次操作”只退出待确认意向，不会取消已提交预约或订单。历史对话若因隐私过滤只能显示“消息内容未保存”，重开页面不会恢复确认按钮；请重新发起并核对预览。已在隔离 Mock/可控模型页面逐点击验证取消订单按钮，本版真实模型页面尚未复验；页面使用前须重启服务。按钮版本校验依赖进程内同会话锁，不代表已完成真实身份认证或多 Worker 并发安全。

文件会话库会在同目录建立 `<会话库名>.checkpoints.db`（例如 `blood_agent.checkpoints.db`）；可用 `BLOOD_AGENT_CHECKPOINT_DB` 指定独立文件。`storage/checkpoint.py` 沿用原 `SqliteSaver` 的 `checkpoints`/`writes` 表与序列化格式，可直接读取旧文件；切换前仍建议备份。服务重启或重新打开页面后，左侧默认显示该用户最近会话；若要恢复更早或不在最近 30 条中的会话，可在调试区填写原用户 ID 和会话 ID，点击“加载此会话历史”。读取历史只操作对话表，不调用模型或业务工具；随后发送**新消息**时，图从相同 ID 的 checkpoint 接续未完成流程。checkpoint 文件权限为 `0600`，**可能包含未脱敏的预约/收货流程状态**，只能用于合成数据本地演示。旧版只用 `MemorySaver` 时产生的待确认状态无法追溯恢复；当前也不支持跨多 Worker 的同会话锁或生产级迁移。

对话落库使用 Agent 自己的 `BLOOD_AGENT_DB`（默认 `./blood_agent.db`），六张 Agent 自有表由 `storage/models.py` 中的 SQLAlchemy 模型和 `AgentRepository` 管理，不读写外部业务表。可通过 `AgentRepository(db_path).list_conversation_turns(user_id, session_id)` 查看同一用户和会话的记录；checkpoint 模型在独立的 `storage/checkpoint.py` 中。目前仍使用 SQLite，MySQL/Alembic、Redis 和跨多 Worker 状态一致性尚未实施。文件库和 checkpoint 仍需按敏感数据管理。

`BLOOD_AGENT_CONVERSATION_RETENTION_DAYS=0` 默认不自动删除现有记录；显式设为正整数并重启后，**下一次保存新轮次时**只清理过期的对话表行及空会话标题/摘要，不清理 checkpoint 或长期偏好。可信本地代码可用 `ConversationGraph.delete_session(user_id, session_id)` 协同清理单会话对话表与 checkpoint；仓储的 `delete_conversation_session()` 和 `delete_user_conversation_history()` 仍只清理对话表。业务表、长期偏好、JSONL 追踪和备份各有独立生命周期；没有登录认证，不向网页/API 暴露删除入口。侧栏仅按所填用户 ID 过滤，**不能用于真实个人资料**。

长期偏好演示话术（需配置可用模型并重启）：先说“以后我更喜欢大学城献血车，请记住”，在新会话问“查看我的网点偏好”；再说“以后改成市中心献血屋，请记住我的网点偏好”，或“请忘记我偏好的献血网点”。只支持偏好网点和上午/下午时段；一次性预约选择、历史记录、收货资料和模型推测都不应写成长期记忆。当前只有本地可控模型回归，真实模型的提取质量及当前页面体验留待用户自行试用。

爱好画像演示话术（需配置可用模型并重启）：说“我平时喜欢徒步，也爱摄影”，答复交付后的后台流程会增量处理本条已保存用户发言；然后问“我有哪些爱好？”。也可以说“我不再喜欢徒步”或“请忘记我所有的爱好”测试撤销。一次性话术如“这次我喜欢爬山”不应保存；含姓名、性别、血型、联系电话、地址或业务记录的整句会被过滤，不送画像模型。画像仅在 general 用户明确查询兴趣/偏好时进入回复提示词；献血点与时段偏好继续由原长期偏好记忆服务处理。Python 可控模型测试已覆盖流程，未调用真实模型、未在当前 Gradio 服务逐点击验收；模型提取质量由用户自行试用。

静态标准问答缓存：在没有活动业务流程的新会话问“现在可以上传图片识别吗？”，唯一完整命中时返回经审核的固定答复，不调用模型和业务工具。配置见 `blood_agents/knowledge/static_faq_cache.json`；个人或实时业务问题不得进入静态缓存。
API `/chat` 未传 `session_id` 时会生成并返回新 ID；后续请求须带上返回 ID。用户身份优先取 `X-User-ID`（本地演示，不等于已实现认证）。

页面“本轮执行摘要”显示实际节点流转、模型模式、处理状态与脱敏工具参数/耗时；原始 JSON 保留在折叠区域。摘要在一轮完成后更新，并非实时进度。可重复成功/失败演示见 [演示步骤与验证范围](docs/DEMO_WALKTHROUGH.md)。

预约完整演示：用 `U1001` 发送“预约明天下午到市中心献血屋”→“确认预约”→“查询我的预约记录”；选择该预约并说“取消明天下午市中心献血屋的预约”→“确认取消”。也可对另一条仍为“已预约”的记录说“签到预约号 XXX”→“确认签到”。切换 `U1002` 查询四状态合成旧数据。网页进程须重启才能加载新版代码；此预约业务无可用模型时只显示服务不可用，未交付有效预览后继续确认不会提交；API/调试结构仍可用于离线验证。

### 按会话追踪执行日志

本地演示默认向**启动服务的终端**输出完整 JSON 行，每条记录有中文 `event_zh` 和稳定的英文 `event`，并用 `user_id`、`session_id`、本轮唯一的 `trace_id` 串起导航、子图、模型、规则纠正、工具与回复。网页的“本轮执行摘要/原始编排数据”和 API `/chat` 响应也返回 `trace_id`。注意：Gradio 的聊天窗口不是服务终端，旧进程不会自动加载新日志配置。

如需留存并按会话查询，在启动服务前设置日志文件（复制新版 `.env.example` 的配置也可以；已有 `.env` 请核对是否仍为 `summary`）：

```bash
BLOOD_AGENT_TRACE_MODE=full \
BLOOD_AGENT_TRACE_FILE=./logs/blood-agent-trace.jsonl \
.venv312/bin/python -m blood_agents.web
```

另开一个终端查看实时日志或最近一轮的中文时间线：

```bash
tail -f logs/blood-agent-trace.jsonl
.venv312/bin/python -m blood_agents.log.trace_viewer --latest --raw
.venv312/bin/python -m blood_agents.log.trace_viewer --session-id '你的会话ID' --raw
.venv312/bin/python -m blood_agents.log.trace_viewer --trace-id '页面显示的本轮追踪ID' --raw
```

重点事件是 `model.request`（`model_input` 为完整输入）、`model.response`（`model_output` 为供应商原始输出）、`understanding.result`（结构化理解）、`appointment.correction`（模型值、原话解析值、最终值）、`reply.result`（最终回复来源）、`tool.finished`、`turn.end`。`stage=navigation/appointment` 是理解调用，`stage=appointment_reply/points_reply/...` 是领域回复调用；`model.attempt.finished` 在 `full` 模式增加脱敏后的 `error_message`、`cause_chain`，HTTP 异常还会记录 `http_status` / `request_id`（如果 SDK 提供）。`timeout_source=gateway_wait` 表示网关等候到期，并非供应商返回了 `TimeoutError`；原请求稍后结束时会另有 `model.attempt.late_result`，其结果不会再用于本轮答复。无模型配置时没有模型请求/响应；模型超时或连接失败时可能没有原始响应，理解会降级为规则，回复会报服务不可用。查看器默认只显示摘要，`--raw` 才展开详细错误。`BLOOD_AGENT_TRACE_MODE=summary` 只记流程元数据，不含异常原文；`off` 关闭追踪。

完整日志会包含用户原话、最近对话、完整模型提示词、原始输出和脱敏后的供应商异常信息；其中仍可能出现个人信息或供应商响应内容。默认完整模式只适合本地演示，部署或真实用户场景请改为 `summary` 或 `off` 并落实脱敏。只有显式配置文件路径时才会落盘，文件以 `0600` 权限创建，`logs/` 已加入 Git 忽略；已知 API Key、Bearer 和常见鉴权字段会从异常文本中剔除，但不要把本地完整日志公开分享。修改日志配置后重启服务。

示例对话：

```text
我要预约
还是上次的网点，有哪些时间？
选择系统返回的日期和时段
确认预约
查询我的预约信息
我的积分和礼品有哪些？
按我的积分推荐礼品
兑换 R001
新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号
确认保存地址
确认兑换
查询我的兑换订单
新增收货地址：收货人张三，手机号13800001234，详细地址上海市人民路100号
确认保存地址
查看我的收货地址
设置默认收货地址 <地址簿返回的地址编号>
查询订单状态
取消订单 <查询返回的订单号>
确认取消订单
查询我的献血记录
献血前可以空腹吗？
```

启动 API：

```bash
uvicorn blood_agents.api:app --reload
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"U1001","session_id":"S001","message":"查询我的献血记录"}'
```

安装 UI 依赖后，FastAPI 模式还会把同一个 Gradio 页面挂载到 `http://127.0.0.1:8000/ui`。

运行测试与评测（当前检索集为 7 条正例、2 条无证据负例，数值不代表生产准确率）：

```bash
pytest -q
blood-agent-eval
.venv312/bin/python -m blood_agents.evals.conversation_evaluation \
  --dataset evals/conversations/batch_review_example.xlsx
# 积分子智能体：3 个独立场景、10 轮固定话术；运行前确认外部模型的数据发送范围
.venv312/bin/python -m blood_agents.evals.conversation_evaluation \
  --dataset evals/conversations/points_lifecycle.xlsx
```

示例 Excel 可批量编辑独立场景、每场景多轮固定话术、场景评审目标和断言。正式评测始终调用被测智能体模型和整场对话评审模型；缺配置、调用失败或规则降级不能通过。评审不核验 Mock 内部业务规则。终端实时显示场景、轮次和模型评审进度；每次运行在 `evals/reports/` 生成以运行 ID 命名的 Excel 报告（另留 JSON 供程序追溯）。详见[自动化对话评测说明](docs/CONVERSATION_EVAL.md)。

## 启用可选演示组件

```bash
.venv312/bin/python -m pip install -e '.[rag]'
VECTOR_BACKEND=faiss .venv312/bin/python -m blood_agents.evals.evaluation
# Milvus Lite 使用本地持久化 .db；首次入库，重启后复用。
GRPC_ENABLE_FORK_SUPPORT=0 VECTOR_BACKEND=milvus RAG_MILVUS_URI=./data/blood_knowledge.db \
  .venv312/bin/python -m blood_agents.evals.evaluation
# 可选：在当前后端上启用 bge-reranker 精排，默认不开启。
RAG_RERANKER=bge VECTOR_BACKEND=faiss \
  .venv312/bin/python -m blood_agents.evals.evaluation
# 扩展挑战集：同一批 18 条正例、10 条负例对比关闭/开启精排；详细结果写入 evals/reports/。
VECTOR_BACKEND=faiss RAG_RERANKER=none \
  .venv312/bin/python -m blood_agents.evals.rag_comparison
# 配好模型后可单独评测当前 40 条问题的证据核验；会产生外部模型调用。
VECTOR_BACKEND=faiss RAG_RERANKER=none \
  .venv312/bin/python -m blood_agents.evals.rag_evidence_evaluation
# 只定向复测一题，仍保留完整数据集的版本和 SHA-256；不把单题结果算成全套成绩。
VECTOR_BACKEND=faiss RAG_RERANKER=none \
  .venv312/bin/python -m blood_agents.evals.rag_evidence_evaluation --case-id P19
```

挑战集当前修订号为 `2026-10-27-public-only`，含 15 条公开资料正例、13 条负例；虚构业务流程正例已移出。旧 40 条报告与当前语料不能直接比较，当前 28 条尚未运行真实模型评测。运行上述证据核验命令会产生外部 API 调用与费用。

FAQ 回复可引用本轮已核验片段中的一个实际使用来源，不必罗列所有候选；无引用、编造链接或标题、引用虚构资料却不说明演示性质仍会触发重生成。该门禁只检查来源边界，不等于已核实自然语言答复的每句话。

首次启用 FAISS 或 Milvus 会下载约 1.3 GB 的 `BAAI/bge-large-zh-v1.5`；显式启用精排还需下载 `BAAI/bge-reranker-base` 的约 1.1 GB 权重。FAISS 每次启动重建内存索引，Milvus Lite 将语料向量保存到本地 `.db`。Milvus collection 名包含语料与模型指纹，相同内容重启直接复用，内容变化新建 collection，不自动删除历史索引；需要由维护者自行清理不再使用的旧 collection。项目配置使用 `RAG_MILVUS_URI`，不要把本地 `.db` 路径赋给 `MILVUS_URI`（该名称会被 pymilvus 自身解析）。上面命令的 `GRPC_ENABLE_FORK_SUPPORT=0` 用于规避本机 macOS 评测时出现的 gRPC fork 告警，其他环境需单独验证。`RAG_DENSE_MIN_SCORE=0.55` 是早期 9 条小样本下设置的初值；当前 28 条挑战集尚未重新跑模型核验。新增的模型证据核验提供独立门槛，但不保证判断正确。默认 `VECTOR_BACKEND=memory`、`RAG_RERANKER=none` 不加载向量/精排模型；配置回复模型的 FAQ 命中仍会增加一次证据核验模型调用。不同精排模型或激活配置的分数范围可能不同，不能把精排分数当作拒答置信度；当前仅本机向量库验证，不代表远端集群或生产业务库已联调。

`deploy/docker-compose.yml` 曾放入 vLLM 和 TGI 的本地生成模型服务样例。TGI（Text Generation Inference）是另一种模型服务容器，模板中位于 `tgi` profile；当前应用没有 TGI 专用客户端，也没有启动或验证该容器。用户已明确本项目使用第三方平台 API，因此这些样例保留作历史资料，不是当前部署步骤或待完成任务。

## 旧业务 API 编排流程

```text
入口传入 user_id（当前本机页面尚无真实登录态认证）
  → Supervisor 判断预约意图
  → ToolRegistry.query_user_profile(user_id) → HTTP/MCP 外部业务接口
  → get_appointment_history(user_id)，分析上次网点偏好
  → get_sites_and_slots(user_id)，展示实际可选网点和时段
  → LangGraph 多轮收集 date/site/time_slot
  → 展示实名信息掩码和预约预览
  → 用户明确确认
  → submit_appointment(payload, idempotency_key)
  → 根据 success/code/message 告知成功或失败原因
```

Agent 通过 `LegacyBloodAPI` 契约和 `LegacyHTTPClient` 请求外部业务服务；不直接访问业务表。本机需要替身服务时可启动被忽略的 `local_business/`。真实公司的字段、认证和错误契约尚未联调，未来接入可能需要调整适配层。

## 19 个工具注册项

| 工具 | 类型 | 作用 |
|---|---|---|
| `query_user_profile` | 查询 | 入口用户的实名资料 |
| `query_points` | 查询 | 积分余额 |
| `list_rewards` | 查询 | 礼品目录与库存 |
| `redeem_reward` | 写入 | 原子兑换 |
| `query_redemption_orders` | 查询 | 当前用户的兑换记录、收货状态 |
| `cancel_redemption_order` | 写入 | 确认后取消未发货订单 |
| `query_shipping_addresses` | 查询 | 当前用户收货地址簿 |
| `save_shipping_address` | 写入 | 确认后新增/修改地址 |
| `set_default_shipping_address` | 写入 | 确认后设置默认地址 |
| `set_order_shipping_address` | 写入 | 确认后填写订单收货信息 |
| `query_records` | 查询 | 献血历史 |
| `query_appointment_history` | 查询 | 历史预约与网点偏好 |
| `query_user_appointments` | 查询 | 按入口用户 ID 查询已提交预约；只读，不启动新预约 |
| `submit_appointment` | 写入 | 用户确认后提交旧预约接口 |
| `cancel_appointment` | 写入 | 用户确认后取消已提交预约 |
| `checkin_appointment` | 写入 | 用户确认后为已预约记录签到 |
| `query_sites` | 查询 | 网点及可预约时段配置 |
| `knowledge_search` | 查询 | RAG 检索 |
| `ocr_extract` | 查询 | 图片文字识别 |

`MCPToolGateway` 是原有进程内发现/调用适配，不是标准 MCP 协议。`business/mcp_server.py` 和 `mcp_client.py` 把礼品、积分、献血记录接成标准本地 stdio Server→HTTP 客户端；其余业务工具仍直连 HTTP adapter。网络 MCP 传输、服务端身份授权、所有工具协议化及 MCP 高并发性能均未实现，不能将其视为生产 MCP 服务。

## 核心难点

### 多智能体路由稳定性

大模型可能输出非法路由或受网络影响。路由层限定意图集合并保留规则降级；现有评测集覆盖口语、复合意图和 OCR 意图。`training/` 已有与当前 Agent 注册表一致的合成分类数据、可复现划分和离线规则基线。用户已决定暂缓 LoRA 微调；没有训练、模型产物或主图接入，不能把规则基线记作微调成绩。

### 多轮状态与流程编排

用户往往不会一次说完预约日期、地点和时段。预约 Agent 先查询身份、历史预约及网点配置，把偏好和已选槽位写入 LangGraph 状态，后续消息优先回到未完成流程。收集完整后先展示经回复模型生成、服务端校验并保留确认资格的预览；若模型回复不可用，紧接着的确认不得调用写接口。当前文件 Mock 库使用独立 SQLite Checkpointer，纯内存 Mock 使用 `MemorySaver`；跨 Worker 同会话锁、过期清理和生产级状态迁移尚未验证。

### 多路召回与排序

固定长度切分容易破坏语义边界，因此当前演示结合标题、段落和滑窗切分。默认字符级关键词召回，显式 FAISS 模式在启动时用 BGE 编码片段、`IndexFlatIP` 入库；显式 Milvus 模式写入或复用本地 Lite collection。两者均以同模型编码带检索指令的查询；向量相似度须先过绝对阈值，再与词面分数融合。可选 bge-reranker 仅对通过原检索门槛的候选精排，输出同时保留 `score`（召回强度）与 `rerank_score`（排序信号）；不能把当前 9 条小样本 Recall@K/MRR 当作完整 RAG 效果。

### 工具调用一致性

LLM 只负责理解和表达，预约与兑换由确定性工具完成。幂等键和数据库事务防止模型重试造成重复预约、重复扣分或库存超卖。

### OCR 稳定性

图片方向、清晰度和版式会影响 OCR。当前仅保留按需加载的 PaddleOCR 优先、缺少该依赖时尝试 Tesseract 的原型；真实图片输入和双引擎效果未验收，OCR 已暂缓，不作为本阶段可展示能力。若恢复开发，还需验证图像质量、字段置信度、人工复核和脱敏。

### 模型平台 API 并发

业务 API 与第三方模型平台 API 解耦。当前网关在单进程内限制主备各自的在途调用；本地已验证主平台 503 后由备用平台完成 `/chat` 导航及回复，也已在 FastAPI 入口验证同会话串行、跨会话并行。真实供应商额度、429、超时、p50/p95 和费用仍需实际平台测试，不使用 vLLM/TGI 本地推理指标描述本项目。

### 隐私与医疗边界

生产系统应从认证上下文读取 `user_id`，落实日志脱敏、传输/存储加密、最小权限与审计。资格预检和知识问答只提供提示，不能替代现场医务判断。

## 简历内容与代码映射

| 简历内容 | 代码位置 |
|---|---|
| LangGraph 工作流编排 | `blood_agents/orchestration/graph.py` |
| 多业务领域智能体 | `blood_agents/domains/` 的五个领域模块 |
| Prompt 意图识别与模型回复（已实现）；LoRA 微调（暂缓） | `blood_agents/orchestration/router.py`, `blood_agents/llm/reply_composer.py`；`training/` 仅保留合成数据和规则基线 |
| 多策略 Chunk、多路召回、重排 | `blood_agents/knowledge/rag.py`, `blood_agents/knowledge/adapters.py` |
| PaddleOCR/Tesseract | `blood_agents/domains/ocr_service.py` |
| 10+ 工具与 MCP | `blood_agents/business/tools.py`, `blood_agents/business/mcp.py`, `blood_agents/business/mcp_server.py`, `blood_agents/business/mcp_client.py` |
| 本地 Milvus Lite 检索（已验证）；vLLM/TGI（非当前范围） | `blood_agents/knowledge/adapters.py`；`deploy/docker-compose.yml` 仅保留未实跑的历史模板 |
| 评测指标 | `blood_agents/evals/`, `evals/` |

## 目录

```text
blood_agents/       智能体、RAG、OCR、MCP、业务层与服务入口
deploy/             旧 vLLM/TGI 与 Milvus Compose 模板，生成模型本地部署非当前范围
training/           合成意图数据、规则基线与暂缓的 LoRA 准备材料
evals/              固定离线评测集
tests/              自动化测试
```

演示用户 `U1001` 初始拥有 1200 积分和两条全血献血记录；`U1003` 有一条 2 治疗量的单采血小板记录，方便验证单位隔离。`U1002` 有六种合成兑换订单状态；网页演示种子还包含一条与“已献血”旧预约相符的合成献血记录，基础测试种子则不含该记录。已有本地 SQLite 文件会增量加列、保留原记录；旧记录缺失的采血机构、血型、献血类型不会凭空补齐。所有规则与数据仅供技术演示，不可直接用于真实医疗、支付或物流决策。
