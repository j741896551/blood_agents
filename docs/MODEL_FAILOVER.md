# 模型 API 容灾与调用链解读

本模块只管理模型请求，不接管旧业务系统，也不保证进程、数据库或整机高可用。

## 阅读顺序

1. `router.py / from_env()`：分别建立主/备 ChatOpenAI 客户端，关闭 SDK 自动重试，读取超时及熔断配置。
2. `router.py / navigate()`：主导航只校验 `NavigationDecision.intent`；进入预约子图后，`interpret_appointment()` 再校验 `AppointmentUnderstanding` 的槽位和动作。
3. `model_gateway.py / call()`：按主备顺序尝试，每服务每轮最多一次；成功响应先校验契约再返回。
4. `graph.py / supervisor()`：先决定领域；预约子图的 `appointment_understand` 独立理解预约输入，再由确定性执行节点调用业务工具。
5. `tool_planner.py / plan()`：预约理解后，模型从受限只读工具中提出请求；业务代码会补查约束所需事实。
6. `reply_composer.py / compose()`：各领域业务结果确定后，独立回复节点复用主备网关生成自然语言；关键事实校验失败会尝试重生成一次。
7. `risk_agent.py / review()`：回复发布前审查；否决后重生成一次，再否决由风控模型拒答。风控服务不可用时只报服务不可用。

因此每个模型调用发生主模型可切换故障时，备用模型处理的是该调用的同一份输入，不会重放预约工具。有模型的一轮预约通常依次调用导航、预约理解、只读工具规划、最终回复、风控审查；被拦截时另有一次重写/复审或安全拒答。它们默认共享同一个进程内网关，不是独立高可用集群。理解契约禁止模型直接返回伪造预约结果或用户身份，回复层只表达已核实事实。

## 配置

如果还没有 `.env`，可从 `.env.example` 创建；已有 `.env` 时直接编辑，不覆盖现有配置。主服务的 Key、接口地址和模型名称是同一组，备用服务使用独立的 `BACKUP_` 前缀：

| 项目 | 主服务 | 备用服务 |
|---|---|---|
| 密钥 | OPENAI_API_KEY | BACKUP_OPENAI_API_KEY |
| OpenAI 兼容接口 | OPENAI_BASE_URL | BACKUP_OPENAI_BASE_URL |
| 模型名称 | OPENAI_MODEL | BACKUP_OPENAI_MODEL |
| 单次等待秒数 | LLM_TIMEOUT_SECONDS（开发默认 40） | BACKUP_LLM_TIMEOUT_SECONDS（开发默认 20） |
| 每进程最大在途数 | LLM_MAX_CONCURRENCY（1） | BACKUP_LLM_MAX_CONCURRENCY（1） |

某服务 Key 留空则不注册；都为空则直接走规则。仅配置备用也可运行。
接口地址留空时使用 OpenAI 默认地址，备用不会继承主服务地址。
接入其他供应商时应明确填写其兼容端点和实际模型名称；非兼容接口需另做适配。
例如 DeepSeek 作为主服务时，设置 `OPENAI_API_KEY` 为 DeepSeek Key、`OPENAI_BASE_URL=https://api.deepseek.com`、`OPENAI_MODEL` 为该平台实际可用的模型名。当前项目仅调用第三方平台 API，不部署本地生成模型；备用地址只由 `BACKUP_OPENAI_BASE_URL` 指定。
不要将真实密钥提交 Git。

- `LLM_TOTAL_TIMEOUT_SECONDS=60`：一次网关调用的整体模型等待预算，耗尽不再尝试备用。
- `LLM_FAILURE_THRESHOLD=2`：连续可切换故障达到阈值后熔断。
- `LLM_CIRCUIT_COOLDOWN_SECONDS=30`：冷却秒数。到期由下一个请求进行一次探测，成功恢复主服务优先，失败重新熔断。

开发阶段当前按主服务最多 40 秒、备用最多 20 秒、一次网关调用整体最多约 60 秒配置；主服务提前报错时备用可立即接续，但备用自身仍最多等待 20 秒。不保证每次错误都恰好等满 60 秒。一次用户对话可能经过导航、预约理解、工具规划和回复等多次网关调用，因此整轮对话可能超过 60 秒。网关总预算不包含后续业务查询时间，也不是整个网页/API 请求的 SLA；交付前需依据真实调用延迟和失败率重新调短并复测。

## 异常与降级

超时、连接失败、429 和 5xx 可以切备用，并计入熔断。认证/权限、其他 HTTP 请求错误、非法输出及内容拒绝不切换；理解节点走现有规则降级，回复节点返回明确的服务不可用错误，避免盲目重试。
回复内容未通过确定性事实校验或风控审查时，`ReplyComposer` 可在业务工具执行完毕后额外重生成一次；这是回复层重写，不是网关对同一供应商的网络重试，也不会再次执行提交工具。
规则降级保留已有槽位，对含糊的“好的”不自动提交；只有已有预览并明确确认才允许原有提交逻辑执行。

`model_info.navigation` 与 `model_info.appointment` 分别记录两个理解阶段，顶层 `attempts` 为带 `stage` 的合并列表；最终回复另用 `response_info` 记录模式、服务和尝试。每次尝试包含 provider、status、error_type、exception_class、duration_ms；provider 为 primary/backup。`invalid_model_output` 表示 JSON/契约校验、空响应或回复事实校验失败，`model_error` 表示其他非 HTTP 客户端异常，`model_refusal` 表示显式拒答。旧版 `invalid_output_or_error` 混合了这些类别，仅凭旧轨迹不能确定具体异常。`circuit_open` 表示冷却中跳过，`busy` 表示已达本地并发上限或恢复探测独占。`model_info`/`response_info` 不包含 API Key、端点、提示词或异常原文；本地 full 追踪日志会单独记录提示词和用户原话，隐私边界见 README。

## 并发与超时限制

同步 SDK 不能强制取消在途请求。网关在预算到达后停止等待，晚到结果不会写入图状态；后台调用仍可能完成并产生供应商费用。
为避免悬挂请求堆积，每个服务有独立的有界在途上限：`LLM_MAX_CONCURRENCY`、`BACKUP_LLM_MAX_CONCURRENCY`，默认各为 1，可设置 1–64。达到上限时标记 `busy` 并尝试备用；超时的底层请求仍占名额，直到后台真正结束。即使把上限调高，熔断冷却后的主服务也只允许一个恢复探测。
不同用户/会话可并行运行模型调用，同一会话仍由 `ConversationGraph` 串行化。所有服务满载时理解节点回退规则、回复节点报服务不可用。上限是单个进程内保护，不是供应商配额；多 Worker 上限会累加。本实现仍**不是经真实平台验证的高吞吐方案**。
生产客户端同时配置 HTTP 超时；自定义客户端若永远不返回，该服务保持 busy，直到服务重启。
熔断状态在网关实例/进程内共享，不跨 Worker，也不会持久化。暂不支持流式部分响应切换。

## 可重复验证

```bash
PYTHONPYCACHEPREFIX=/private/tmp/blood-agent-pycache .venv/bin/python -m pytest tests/test_model_gateway.py -q
.venv/bin/ruff check .
```

测试覆盖主服务成功、同输入主备切换、实际 ChatOpenAI HTTP Mock 的 429/503/401/400、非法输出与拒绝、熔断与成功/失败探测、并发探测、在途保护及总预算。完整 LangGraph 测试验证降级后保留预览、不确认不提交、恢复后提交仅一次。
自动化测试均为本地模拟。此前曾用主服务配置对“你好”和一条合成预约语句完成两次理解契约调用；本次回复节点改造后尝试真实模型端到端问候，当前执行环境中主备均报 `APIConnectionError`，故**不能**把该试跑记为真实回复通过，也无法据此判断供应商是否故障。真实供应商回复质量、主备联调、可用率和线上容量尚未验证。
