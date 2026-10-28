# 智慧献血多智能体代码解读

本文建议按照“入口 → LangGraph → 领域 Agent → 旧业务 API → 数据层”的顺序阅读，而不是按文件名逐个阅读。

当前实现目录：`orchestration/` 管主图、注册和路由；`domains/` 管各领域 Agent；`llm/` 管模型网关、规划、回复和风控；`business/` 只管外部业务契约、HTTP/MCP 适配及工具；`storage/` 管 Agent 会话与偏好；`knowledge/` 管检索。实际业务表、种子和服务端在同级、被 Git 忽略的 `local_business/`，Agent 不直接访问这些表。以下路径除特别注明外均相对 `blood_agents/`。本文部分历史样例提到本地替身，是测试解释，不表示这些代码仍位于 Agent 包。

阅读函数时，先看紧跟 `def` 的中文 docstring：它概括“这个函数做什么”；再看函数内的行内注释：它解释“为什么要这样限制、何时不能写入”。例如 `ConversationGraph.invoke()` 管一轮会话与状态恢复，`appointment_agent()` 根据接口数据推进预约，`ReplyComposer.compose()` 只负责基于本轮事实组织答复。测试用例的函数名通常直接描述验证场景；本次逐函数 docstring 覆盖运行代码与 `training/` 辅助脚本，没有机械地给每条测试用例加重复说明。

## 1. 一条消息如何进入系统

系统有三个入口：

- `blood_agents/web.py`：Gradio 网页。
- `blood_agents/api.py`：FastAPI `/chat` 接口。
- `blood_agents/cli.py`：命令行。

三者最终都会把本轮输入交给 `ConversationGraph.invoke()`。下面是便于理解的示意调用，不是三个入口逐字相同的源码：

```python
graph.invoke(
    {"messages": [HumanMessage(content=message)], "user_id": user_id},
    config={"configurable": {"thread_id": session_id}},
)
```

`user_id` 表示入口用户，生产环境应由认证网关注入；入口的 `thread_id` 值是页面/API 使用的 `session_id`。`ConversationGraph` 会把 `(user_id, session_id)` 编码为图内部 checkpoint 键，因此同一用户同一会话记住上一轮参数，不同用户不共享。每一轮还会生成新的 `trace_id` 用于关联日志，它不是会话 ID。Gradio 的 `history` 只负责页面展示，不能用它恢复待确认操作。详见第 12 节。

Gradio 的侧栏标题另走展示元数据流程：`ConversationGraph.invoke()` 先完成答复并把脱敏问答存入 `conversation_turns`；`web.py` 的首轮回复事件先把正文交给页面，随后触发 `refresh_session_title()`。它从仓储重新读取该用户和会话的已脱敏轮次，`llm/conversation_summarizer.py` 用模型生成结构化标题与摘要，仓储再保存到 `conversation_session_summaries`。侧栏只读保存的标题，不再截取首句；旧会话被打开时才按需补生成。标题模型失败只留下“待生成标题”，不会重跑业务图或改变答复。`summarized_through_turn_id` 防止迟到的摘要覆盖更新轮次。标题摘要不作为 Agent 记忆；短期任务快照、长期网点/时段偏好、后续第 13 节介绍的爱好画像和静态标准问答缓存各自走独立路径。

## 2. LangGraph 如何路由

`blood_agents/orchestration/graph.py` 中的 `build_graph()` 创建一个主图，并把各领域编译子图作为节点装入主图：

```text
START
  ↓
static_faq_cache ── 完整命中经审核的标准问答 → END（不调用模型/业务工具）
  │ 未命中
  ↓
supervisor（只识别领域意图）
  ├─ appointment_subgraph
  │    appointment_subgraph_start → appointment_understand → appointment_tool_plan
  │    → appointment_agent → appointment_reply → appointment_handoff
  │    ├─ 未结束：保存 active_agent，等待下一轮
  │    └─ 终态：main_agent_return → END
  ├─ points_subgraph → points_understand → points_agent → points_reply → points_handoff
  │    ├─ 待确认：保存 points_state，等待“确认兑换”
  │    └─ 完成/退出：main_agent_return → END
  ├─ records_subgraph ┐
  ├─ ocr_subgraph     ├─ 各自处理 → 领域回复节点 → main_agent_return → END
  ├─ faq_subgraph    ─┘
  └─ main_agent_response → general_reply → END（问候、能力询问等）
```

`static_faq_cache` 只在完整命中经审核的静态问答时直接结束本轮；未命中才进入 `supervisor`。`supervisor` 调用 `IntentRouter` 获取本轮意图，再由主图条件边派发到相应子图。`blood_agents/orchestration/agent_registry.py` 的 `AgentSpec` 一次定义意图、能力描述、节点工厂、规则匹配和流程类型；`AgentRegistry` 向主导航生成能力清单和允许的意图，向 `build_graph()` 提供子图注册与条件边映射。主导航自己的 `main_agent_response` 处理问候、能力询问和非领域请求，不调用 FAQ 知识检索。预约活动期间，含糊的补充信息默认续接预约；明确的积分、记录、献血知识问题等仍可切换到对应子图。记录、OCR、FAQ 一次性子图在本轮结束后交给 `main_agent_return`；预约和积分子图可等待下一轮。预约中临时查询积分后仍可继续原预约。

`build_graph()` 中会创建多个 `StateGraph(AgentState)`：一个主图和预约、积分、记录等子图。它们在服务启动时编译，用户发消息时才运行节点。子图 `compile()` 后挂入主图；主图统一安装 checkpointer，所以“子图结束”不等于“用户流程结束”。只有领域交接状态表明等待用户或返回主图，下一轮才会按 checkpoint 中的 `active_agent`/领域状态续接。

### 用一轮预约请求串起代码

以“我想预约明天下午的献血”为例，建议按下面顺序打断点阅读：

1. `web.py:ChatApplication.respond()` 或 `api.py:chat()` 取入口 `user_id` 和 `session_id`，只传本轮消息；`orchestration/conversation.py:ConversationGraph.invoke()` 加会话锁、生成 `trace_id`，再调用图。
2. `graph.py:cache_lookup()` 检查是否完整命中静态问答；未命中时 `supervisor()` 通过 `IntentRouter.navigate()` 得到 `appointment`。这里的模型只决定去哪个领域，不直接生成预约回复。
3. 预约子图的 `appointment_understand()` 提取日期/时段等意向；`appointment_tool_plan()` 可让模型提出白名单内的只读工具；`domains/appointment.py:appointment_agent()` 再使用查询结果或补查必需事实，核对真实可约组合并更新跨轮槽位。此时只形成待补充信息或待确认预览，不会因为模型说“已预约”就写库。
4. `appointment_reply` 把本轮 `result`、已选参数和可选时段交给 `ReplyComposer.compose()`；模型负责措辞，代码检查关键事实与隐私边界。`appointment_handoff` 决定是等待用户下一轮补充/确认，还是返回主图。
5. `ConversationGraph.invoke()` 保存脱敏对话轮次并返回状态。排查时用同一 `trace_id` 对照 `node_trace`（走了哪些节点）、`tool_trace`（调用了哪些工具及后端）、`model_info`（理解模型/降级）和 `response_info`（回复模型/校验结果）。对话轮次供历史展示，checkpoint 供恢复流程，两者不能互相重放。

如果下一轮用户明确确认预约，执行节点才调用写工具；是否成功和失败原因以接口回执为准。积分兑换、地址修改和取消订单也遵守“查询事实 → 展示预览 → 用户确认 → 写工具回执 → 模型表达”的边界。

问候的内容来自 `main_agent_response.result.capabilities`，最终措辞由 `general_reply` 调用模型。`llm/reply_composer.py` 的 general 提示词要求把当前能力按主题写为 Markdown 项目列表；若首稿写成连续段落，回复节点带着同一组业务事实让模型改写一次。格式改写失败会保留已通过事实校验的首稿，同时记录质量提示，不会因为排版问题返回服务不可用。Gradio 只渲染模型给出的 Markdown，不负责把段落转换成列表。

要追踪一条消息，从 `ConversationGraph.invoke()` 看起：入口的 `user_id`、原始 `session_id` 和本轮 `trace_id` 被写入 `blood_agents/log/trace_logging.py` 的 `ContextVar`，因此同轮的导航、预约语义节点和工具日志都有相同标识。`ModelGateway` 在后台线程调用模型时复制该上下文，主备服务的原始输出仍可归到正确的用户和轮次。`summary` 模式只记录流程；`full` 模式记录用户原话、完整模型输入/输出，以及 `appointment.correction` 的校正前、原话解析和校正后槽位。终端输出和可选 JSONL 文件的启用方式见 README。日志显式记录用户 ID，不主动复制实名资料或提交载荷；完整模型上下文仍可能包含对话中的个人信息。随机生成的 `trace_id` 保持原样，以免数字片段偶然命中手机号正则而丢失关联；用户文本及其他日志字段仍脱敏。

每轮消息先进入 `IntentRouter.navigate()`。配置大模型时先尝试 LLM 结构化路由；模型失败或没有 API Key 时使用注册表中的规则匹配降级。规则只把明确的献血知识问题归到 `faq`，不再把所有未知消息都交给知识库。主导航提示词在 `blood_agents/orchestration/router.py` 的 `SUPERVISOR_PROMPT`，能力列表从注册表插入，JSON Schema 的意图枚举也由同一注册表生成，其中 `general` 表示问候或非领域请求。2026-10-20 的修订用例真实模型评测中，主导航两轮均走 `general`，但只有小样本，不能据此推断总体识别准确率。

新增单轮领域 Agent 时，先实现执行节点，再在 `default_agent_registry()` 注册 `AgentSpec`：能力描述供导航提示词和问候展示，`rule_match` 供无模型降级，`node_factory` 供主图构建子图。注册表在主图编译前冻结，运行中新增能力必须重建图；未知模型意图无法通过当前运行时生成的 schema。预约专属多轮状态机仍在 `graph.py`，不能把其他多轮业务仅标成 `appointment` 就自动获得相同流程。此注册方式不是自动扫描 Python 文件，也不是 MCP 工具发现。

### 子 Agent 的系统提示词在哪里？

五个领域子图中的业务执行节点分别位于 `blood_agents/domains/` 的 `appointment.py`、`points.py`、`records.py`、`faq.py`、`ocr.py`。预约子图有 `appointment_understand` 模型理解节点，使用 `blood_agents/orchestration/router.py` 的 `APPOINTMENT_PROMPT` 提取槽位和动作；主导航使用独立的 `SUPERVISOR_PROMPT`。各领域执行节点只返回结构化事实与状态，随后由独立回复节点（例如 `appointment_reply`、`faq_agent_reply`）使用 `blood_agents/llm/reply_composer.py` 中的领域提示词生成用户答复；不是让模型重新查询或提交业务。FAQ 无检索证据时仍调用模型，但明确要求只解释证据不足、不得补造医疗知识。

预约子图在理解之后新增 `appointment_tool_plan`：`blood_agents/llm/tool_planner.py` 的模型提示词只允许列出 `query_sites`、`query_appointment_history`、`query_user_profile`、`query_user_appointments` 四个只读工具名。图节点用入口 `user_id` 执行，模型输出不能指定身份或提交参数；结果交给 `appointment_agent` 使用。缺少业务约束必需的事实时，执行 Agent 会补做查询，因此“模型规划”不等于允许模型绕过资料、可用时段或用户确认校验。无模型或规划失败时仍可在测试中观察确定性执行，但回复只报服务不可用。

回复节点保留了 `blood_agents/llm/risk_agent.py` 的独立内容风控 Agent 接线，但当前默认 `BLOOD_AGENT_RISK_MODE=passthrough`：不请求风控模型，记录透传状态后发布已通过确定性关键事实校验的正文。可显式设置 `model` 审查违法有害、辱骂、歧视、威胁和危险个体化医疗建议；风控模型输出 `allow/reason`，否决后最多重写一次，仍否决则由风控模型写拒答并再次审查。它不负责业务字段逐字匹配，也不是独立 LangGraph 子图。当前透传模式不具备模型内容风控能力；即便开启模型审查，也不能当作生产安全或法律合规证明。

目前理解和回复提示词都集中在代码中，尚未外置为可编辑配置文件；提示词配置化仍需单独确认范围。身份、可用时段、确认和写操作边界继续由代码负责。回复模型默认复用现有主备网关；无 Key 或模型服务失败时只报服务不可用。校验只硬拦无回执成功宣称、关键结果缺失和隐私泄露等；接口内部提示的同义改写不再硬拦，普通质量遗漏只记告警。完整手机号/证件号先脱敏再复检；两次关键校验仍不通过时默认透传模式报服务不可用。`response_info` 标注来源、质量提示和风险模式。业务节点不再拼接对用户的底稿。

## 3. AgentState 中保存了什么

`blood_agents/core/models.py` 定义共享状态：

- `messages`：对话消息，使用 `add_messages` 自动追加。
- `user_id`：当前登录用户。
- `intent`：Supervisor 的路由结果。
- `active_agent`：跨轮标记当前尚未结束的领域流程，目前用于预约子图。
- `appointment_state`：预约子 Agent 的权威流程状态；兼容字段 `slots` 同步给现有页面和 API 展示。
- `agent_status` / `agent_handoff`：领域子图标记“等待用户”或“完成”，并记录控制权交给用户还是返回主图。
- `result`：领域 Agent 返回的结构化业务结果。
- `tool_plan_info`：预约子 Agent 本轮模型请求的只读工具与规划模式；即使规划失败，执行 Agent 仍可能补查必需事实。
- `response_info`：本轮最终回复来自模型、风控拒答，或因无模型、调用失败、事实/风控校验失败而不可用。
- `retrieved_context`：RAG 检索结果。

预约状态大致会这样变化：

```text
{}
→ {flow: appointment, status: collecting, preferred_site: 市中心献血屋}
→ {date: 2026-10-01, site: 市中心献血屋, time_slot: 上午}
→ {status: awaiting_confirmation, donor_kind: previous, donor_confirmed: false}
→ 用户答“是”：{donor_confirmed: true}，仍未提交
→ 用户答“确认预约”：本轮调用接口，按真实返回成功或失败
→ {}
```

完成或取消后 `appointment_state` 与 `slots` 都清空，`active_agent` 置空，交接结果标记 `control=main`。预约未完成时保留状态并设置 `active_agent=appointment`；临时查询积分时仍保留预约状态。

## 4. 预约 Agent 的核心流程

`blood_agents/domains/appointment.py` 中的 `appointment_agent()` 是编排层，不负责生成预约结果。

### 4.1 查询用户和业务配置

进入预约流程后调用：

```python
legacy_api.get_user_profile(user_id)
legacy_api.get_appointment_history(user_id)
legacy_api.get_sites_and_slots(user_id)
```

用户资料用于检查登录用户身份；历史预约提供上次网点及上次献血人的演示数据；网点配置决定实际可选择的日期和时段。登录用户是代办人，不一定是本次献血人。

### 4.2 收集槽位

Supervisor 调用 `IntentRouter.navigate()`，用 `NavigationDecision` 只校验领域意图；进入预约子图后，`appointment_understand` 调用 `IntentRouter.interpret_appointment()`，将当前日期、已收集槽位、上一轮可用网点及最近消息作为上下文，用 `AppointmentUnderstanding`/`AppointmentSlots` 校验参数和动作。两次模型调用共用同一主备网关，失败分别降级，`model_info.navigation` 和 `model_info.appointment` 分开记录。

规则模式使用 `router.py` 中的 `_rule_appointment()` 解析日期、上午/下午和网点；“还是上次网点”使用历史接口返回的 `preferred_site`。模型超时、非法 JSON、非法日期或额外身份字段会降级为规则模式。若整个模型回复是一个完整的 `json` 代码块，先去掉围栏再按同一 Schema 校验；代码块前后夹杂解释文字仍拒绝。脱敏轨迹区分输出校验失败与其他错误。模型不能填写 user_id、流程状态或预约结果。

模型返回合法 JSON 但漏掉“明天/后天”时不会自动降级。因此 `interpret_appointment()` 会再从用户本轮原话中提取明确且无歧义的日期/时段，并用上一轮查询接口返回的网点名核对明确提及的网点；若与模型槽位不同，以用户明确表达为准，在 `model_info.appointment` 标记校正。它只校正输入，不决定网点是否可预约。比如先选“明天下午”，下一轮改问“后天可以约吗”但没有重选时段，预约 Agent 会清除旧“下午”，展示该网点后天实际可选时段，等待用户选择；不会沿用旧时段，也不会自动提交。

当前轮参数与已保存参数不同，会回到收集状态，重新生成预览；即使同一条消息同时说“修改并提交”，也要等待下一轮确认。规则模式仅接受明确的提交词，模型模式通过 action 表示确认，但仍要求上一轮已经生成预览。

信息不完整时，Agent 返回 `need_more_info`，同时把已获得参数保存在 `slots` 中。`appointment_reply` 读取已选参数、历史偏好、缺失项和接口可选组合，由模型自然表达；两个及以上网点时提示模型用 Markdown 表格并列显示网点及其时段。模型不可用或遗漏关键事实时只报服务不可用，不把结构化结果自动拼成业务答复。如果日期已确定，可选网点/时段只传入该日期的接口结果，避免混入其他日期。“明天”只确定日期，仍需用户选择上午/下午及网点，不能由 Agent 自动代选。

### 4.3 校验和确认

信息完整后，`_find_available_slot()` 检查日期、网点和时段组合是否真的存在于接口返回值中。校验通过后只返回预约预览，附上上次献血人的姓名、掩码身份以及献血类型/献血量，并询问是否为本次献血人；代他人预约须由用户明确提供姓名、手机号、身份证号、性别、血型及献血类型/献血量。状态为 `awaiting_confirmation`，此时还没有写数据库。仅回答“是/对”会标记献血人已确认，但仍要求最后明确确认整个预约；“确认预约”一次性确认网点、时段和献血人。

### 4.4 提交预约

只有用户下一轮明确回复“确认预约”等确认词，Agent 才调用：

```python
legacy_api.submit_appointment(payload, idempotency_key)
```

Agent 在确认轮直接等待 Mock 接口返回，再把 `success`、`code`、`message` 及成功时的真实预约号送给回复模型；没有“已经提交但等待回执”的中间成功状态。Mock 把献血人姓名和掩码手机号写入 SQLite；同一献血人同一天重复预约返回 `ALREADY_BOOKED`，时段满额返回 `SLOT_FULL`，幂等重试返回原预约号且不再次扣余量。这些只是演示规则，不代表生产业务资格校验。回复模型不能将未提交预览说成等待接口回执。

### 4.5 查询已提交预约

“查询我的预约信息”仍交给预约子 Agent；`appointment_agent()` 通过 `query_user_appointments` 工具按入口 `user_id` 请求外部业务接口。Agent 不读取业务数据库；回复节点只转述接口回执中的预约信息，也不会因为查询而提交新预约。

### 4.6 退出意向、取消记录与签到

“退出预约”在存在未提交预览时只清空该会话的预约意向，返回 `scope=draft`；它不修改 SQLite 中已预约记录。“取消后天上午大学城献血车的预约”或“签到预约号 XXX”先按入口 `user_id` 查询；必须匹配唯一一条 `booked` 记录，才进入 `awaiting_action_confirmation`。此轮只展示预览，不调用写工具。下一轮“确认取消”/“确认签到”才调用 Mock 对应接口；接口在 SQLite 事务中验证状态和当前用户，并返回成功、重复操作或状态拒绝回执。签到/献血后取消会被拒绝。`donated` 只存在于合成旧数据，没有对话写入入口。最终模型回复只能依据本轮回执或查询结果措辞，不能凭预览宣称已操作。

## 5. 为什么要有 LegacyBloodAPI

`blood_agents/business/legacy_api.py` 定义了协议：

```python
class LegacyBloodAPI(Protocol):
    def get_user_profile(...): ...
    def get_points(...): ...
    def get_appointment_history(...): ...
    def get_user_appointments(...): ...
    def get_sites_and_slots(...): ...
    def submit_appointment(...): ...
```

当前 Agent 只保留 `LegacyBloodAPI` 协议和 `LegacyHTTPClient`。本机替身实现位于被 Git 忽略的同级 `local_business/`；接入真实公司接口时应以其实际认证、字段和错误契约校准客户端，不能预先保证零改动。

这就是依赖倒置：Agent 依赖业务契约，而不依赖 HTTP、数据库或某个具体厂商。

## 6. 为什么第一次就有上次预约

本机替身的 `local_business/api.py` 为 `U1001` 预置了历史偏好，只用于没有旧系统时的本地测试；Agent 侧只能通过查询接口看到该资料。

用户确认后由外部业务接口处理新预约；本机替身把结果写进自己的 `BLOOD_AGENT_LOCAL_BUSINESS_DB`（默认 `./local_business.db`），Agent 自身的 `BLOOD_AGENT_DB` 只保存会话和偏好。首次看到“上次网点”不表示当前已有已提交预约。

## 7. 积分、兑换订单与收货地址

主导航将积分问题交给积分子图。`points_understand` 用模型区分余额查询、礼品目录、兑换意向、确认和退出；礼品编号只取用户本轮明确给出的值，或由 `points_agent` 将本轮明确提到的完整礼品名称与刚查询的目录唯一匹配。`points_agent` 先按入口 `user_id` 查询积分、目录与地址簿。无地址时进入地址收集；地址保存有独立预览和“确认保存地址”写操作。保存成功后才生成包含礼品、积分成本、收货人、掩码电话和详细地址的完整订单预览。已有默认地址时直接把它作为待确认预览的候选；用户也可在最终确认前改选自己地址簿中的地址。`points_reply` 必须让模型成功交付每次预览，否则 `points_handoff` 清除待确认状态。只有用户在完整订单预览后另行回复“确认兑换”，且导航、子图理解均由模型完成，才调用 `redeem_reward`；服务不可用时不调用兑换写工具。

同一积分子图还处理兑换记录、收货状态与地址：`query_redemption_orders` 返回当前用户订单，`query_shipping_addresses` 返回掩码手机号的地址簿；余额和库存筛选可兑换推荐，不由模型捏造礼品。新增/修改地址、设置默认地址、为订单填写地址、取消未发货订单均先读接口事实并展示待确认预览，再按对应的明确确认词调用写工具。订单号和地址编号须来自本轮用户明说并属于当前用户；手机号由服务端从本轮原话提取，完整值仅保存在本地 Mock SQLite，模型输入与回复使用掩码。地址修改可只提供要变更的字段，Mock 层保留其他字段。

本机替身的兑换事务位于 `local_business/repository.py`，不属于 Agent 包。Agent 只负责收集和确认数据、调用接口并根据回执回复；旧业务积分和库存结算不是本项目主线。

1. 用幂等键查询是否已经处理。
2. 校验用户、礼品，以及所选收货地址是否完整且属于入口用户。
3. 复核积分和库存；不足时返回失败原因。
4. 扣除积分和库存，写入含收货信息快照的兑换流水。
5. 在一个事务中提交。

任何步骤异常都会回滚，避免只扣积分但没有生成兑换记录。

新兑换必须有有效收货地址，成功后的订单快照进入“待发货”；没有地址时不扣积分、不生成新订单。历史旧订单可保留“待填写收货信息”状态，仍支持为这类记录补地址。最终接口可能因积分/库存变化或地址失效拒绝下单；领域节点将失败回执交给回复模型说明，不能自行宣布成功。Mock 取消仅允许“未支付、待填写收货信息、待发货”三种状态；已发货/运输中/已收货不可取消，接口返回失败原因。`U1002` 的六条合成旧订单只用于状态展示，没有支付和物流状态推进能力。真实供应商模型及 Gradio 页面新版多轮仍须单独验收。

### 7.1 献血记录怎样查询

主导航识别“查询我的献血记录”后进入记录子图。`blood_agents/domains/records.py` 只使用入口 `user_id` 调用 `query_records` 工具，经 HTTP 或 MCP 查询外部业务接口。`records_found`、`no_records` 与 `records_query_failed` 必须区分；Agent 不直接读取个人记录表。

Mock 记录按日期、献血点、采血机构、献血类型、单次量及单位、ABO/RhD、演示编号与记录关联积分组织。旧库自动增量加列，原来没有的字段保持未知，不从账号资料或别的记录推断；执行节点只把非敏感字段白名单交给回复模型。全血以 mL 统计，单采血小板按治疗量统计，不跨单位相加。`MOCK-D` 是本地演示编号，积分是 Mock 记录关联值，不是电子献血证编号或账户余额。`U1001` 提供两条全血样例，`U1003` 提供一条单采样例；网页种子中的 `U1002` 还有一条与“已献血”旧预约一致的合成记录。所有个人记录来自 Mock，不是网上查询的真实献血数据。当前新增字段已通过本地回归，真实模型新版回复与页面复验仍待完成。

## 8. RAG 问答

`blood_agents/knowledge/rag.py` 的流程为：

```text
多策略切分
→ 关键词召回
→ 可选 bge 向量召回（FAISS/Milvus）
→ 可选 bge-reranker 精排
→ 返回文本、分数和来源
```

默认不下载向量模型，只使用离线关键词召回。`adapters.py` 中提供 BGE、FAISS 和 Milvus 的可选适配器；Milvus 当前只在本机 Lite 模式验证，不称为生产部署。

个人“我下次什么时候可以献血”属于组合问答：FAQ 先检索并核验 `GB 18467-2011` 的间隔片段，再由 `DonationIntervalPlanner` 以结构化输出判断是否需要当前登录用户的献血记录。需要时复用 `records_agent` 和 `query_records` 工具；服务端选最近一条类型/日期完整记录，按规则计算可献全血与机采的最早参考日期。用户明确指定一种类型时，只返回该类型；记录缺失、查询失败或最新记录字段未知时，不计算个人日期。最终回复模型收到本轮证据、查询状态和计算结果，只负责自然表达，并须引用来源。最短间隔日期不是献血资格结论。

兑换订单由 Mock SQLite 返回统一字段：订单号、礼品、积分、状态、收货人、掩码电话、收货地址、下单/付款/发货/签收/取消时间。回复节点使用固定列顺序的 Markdown 表格；缺失字段显示“未提供”。旧表通过增量加列保留原数据，六种演示状态另有合成地址和时间线供页面展示。

## 9. Gradio 页面

`blood_agents/web.py` 中：

- `ChatApplication` 持有数据库和编译后的会话图。
- `respond()` 把网页输入转换成 LangGraph 消息。
- `build_ui()` 只负责组件布局与按钮事件。

业务逻辑没有写在按钮回调中，因此 CLI、API 和网页使用完全相同的智能体流程。

## 10. 推荐阅读顺序

1. `models.py`：先理解状态字段。
2. `orchestration/graph.py` 与 `orchestration/router.py`：理解节点接线与意图判断。
3. `domains/appointment.py`：重点阅读多轮预约；其余四个领域分别在同目录。
4. `business/legacy_api.py` 与 `business/legacy_http.py`：阅读业务契约和 HTTP 客户端。
5. `storage/models.py` 与 `storage/repository.py`：先看六张 Agent 自有表的 SQLAlchemy 映射，再看会话、摘要、偏好和画像的 ORM 事务；旧业务事务仅在被忽略的 `local_business/repository.py`。
6. `web.py` / `api.py`：理解系统入口。
7. `llm/reply_composer.py`：理解模型拿到的事实和发给用户前的校验。
8. `knowledge/rag.py` / `knowledge/adapters.py`：理解知识检索。
9. `tests/test_system.py`：通过测试用例观察预期行为。

## 11. 调试建议

Gradio 页面底部的“本轮编排信息”会展示 `intent`、`result` 和 `slots`。学习预约流程时可以观察 `slots.status` 如何从 `collecting` 变成 `awaiting_confirmation`，最终成功后清空。

运行测试：

```bash
source .venv312/bin/activate
pytest -q
```

调试时优先给关键节点打断点：

- `orchestration/graph.py` 的 `supervisor()`。
- `domains/appointment.py` 的 `appointment_agent.node()`。
- `business/legacy_api.py` 的 `submit_appointment()`。
- `web.py` 的 `ChatApplication.respond()`。

## 12. 会话隔离与工具追踪（2026-10-10）

入口调用链：`web/api/cli → ConversationGraph.invoke → LangGraph Supervisor → 领域 Agent → ToolRegistry.invoke → Mock API/现有本地组件`。

FAQ 从 `knowledge/corpus.json` 白名单加载 11 篇公开资料人工摘要（10 个来源），本机替身的虚构流程资料不再进入默认语料。默认 `VECTOR_BACKEND=memory` 使用词面检索；可选 `faiss`、`milvus` 与 BGE、CrossEncoder 精排仍按配置加载。来源分数只用于检索/排序，模型证据核验仍须判断具体问法是否有直接依据。

`evals/rag_challenge.json` 当前为 14 个来源、40 条问题的人工标注挑战集（27 正/13 负）；用例中的空来源表示静态知识库不能直接回答，而不表示问题与某篇边界说明毫无词面关联。旧版有 28/29 条问题，历史 `evals/reports/rag_*.json` 只保留对应版本的结果。2026-10-20 **旧版题目**在本机 FAISS+BGE 上的 18 条正例 Recall@4 都为 1.0，精排使 MRR 从 0.9722 升至 1.0；10 条负例的无证据率两组都只有 0.2，其中 7 条需要实时或个人事实的问题都误命中静态片段。这说明原有 `grounded` 只表示“检索到片段”，**不等于片段足以回答用户问题**。新版公开资料增加献血筛查、全血/单采、2026 间隔与护理、血液检测、知情同意、电子证、血费减免和表彰；其召回与真实模型证据核验需重新评测。

现在 FAQ 的检索结果若非空，会进入 `knowledge/evidence.py` 的独立核验阶段：模型只返回 `supported` 和支持问题的 `doc_id` 列表，服务端校验编号来自本轮候选后，只把对应原始片段送到回复节点。模型判为不支持时 `result.status=no_evidence`，并记录原始命中数；模型输出非法、超时或不可用时 `result.status=evidence_check_failed`，不把未经核验的候选交给回复模型，也不谎称知识库没有答案。无检索命中则跳过核验。该阶段默认复用已配置的主备模型网关，会增加调用与延迟；离线测试可显式注入核验模型或关闭以隔离旧回复测试。`rag_evidence_evaluation` 是真实模型挑战集评测入口，未配置模型即报错，不以离线规则冒充模型效果。2026-10-20 初版真实模型核验为 16/18 正例、10/10 负例；P10/P14 原题的证据范围不严谨。b 版真实模型核验为 17/18 正例、10/10 负例，P12 原题也有标注边界问题。c 版仅对 P12 做了 1/1 定向真实模型复测，未重跑整套 28 题。当前挑战集已修订为 `2026-10-25-official-expansion`；新版尚未运行真实模型评测。两种评测器的报告均记录数据集修订号与 SHA-256 指纹；`--case-id P12` 只筛选执行题目，不改写完整数据集指纹。来源判定仍不验证最终答复质量，也不能声称消除幻觉或保证医学事实可靠性。

FAQ 最终回复的引用门禁在 `llm/reply_composer.py`：多篇证据被核验通过时，模型只须引用实际用于答复的至少一篇本轮来源，可用原始 URL/`demo://` 标识或资料原文标题；不强迫逐字罗列所有候选来源。回复若没有引用、引用本轮证据以外的链接/标题，或把引用的虚构演示资料说成真实制度，会被拒绝并尝试重生成。这是来源结构边界，不是答案逐句事实核对；2026-10-20 修订版 4 场景/6 轮真实模型评测虽全部通过，人工复核仍发现无证据的个人用药答复附加了资料外医学评估细节，见 `PROJECT.md`。

可选 MCP 链路：先启动外部业务 HTTP 服务，本机无旧系统时可运行 `python -m local_business.http_service`；再设置 `BLOOD_AGENT_BUSINESS_TRANSPORT=mcp`。Agent 自动启动包内 `business/mcp_server.py` 的 stdio 子进程，三个只读工具转发礼品、积分和献血记录查询；其余业务工具经 HTTP 客户端调用。默认 transport 为 `http`，`local` 仅为本机替身兼容测试，MCP 不暴露写操作。

`conversation.py` 将入口 user_id 和 session_id 联合编码再取摘要作为 checkpoint 的 thread_id。同名会话在不同用户下互不共享；同一会话加锁串行更新，防止并发修改/确认覆盖状态。此机制仅用于本地隔离，不能替代真实认证；MemorySaver 和锁暂无过期清理，也不跨进程共享。

文件型 checkpoint 由 `storage/checkpoint.py` 的 `SQLAlchemySqliteSaver` 实现 LangGraph `BaseCheckpointSaver` 接口，独立的 SQLAlchemy 引擎读写 `checkpoints` 和 `writes` 两张表；表结构、二进制序列化和版本号保持与原 `SqliteSaver` 3.0.3 一致。图每轮通过 `get_tuple` 取回流程状态，节点运行时用 `put_writes` 保存待处理写入，轮次结束用 `put` 保存新版本；`list` 供历史版本读取，`delete_thread` 供可信单会话清理。`ConversationGraph.close()` 释放引擎连接池，不删除文件。原文件中的预约待确认预览可在切换后继续确认，测试使用临时合成库；项目真实文件未被迁移或清空。SQLite 和进程内会话锁仍不提供跨 Worker 一致性。

`orchestration/graph.py` 用 `traced()` 包裹领域节点，每轮创建独立轨迹，并以 ContextVar 隔离并发上下文。`business/tools.py` 只记录工具名、白名单业务参数、状态、错误类别/业务码及耗时，不复制用户实名资料、身份、密钥和原始异常。工具错误不自动重试，尤其不重放预约提交。

Supervisor 不再强制把所有消息路由回预约。明确积分/记录查询可暂时切换领域，保留预约槽位；随后“确认预约”继续原预览。取消会清空槽位；参数修改后必须生成新预览。规则对含糊确认保持保守，真实模型理解效果仍需单独验证。

Gradio 的会话 ID 使用加载函数生成，不再在 UI 构建时给所有浏览器同一个 ID；切换用户清空界面历史。API 没有提供会话 ID 时生成新 ID 并返回，客户端必须保存它才可续聊。网页 JSON 与 API 响应已包含 `tool_trace`，更易读的节点/工具摘要属于下一项展示任务。

回归见 `tests/test_conversation_tools.py`，包含图级隔离/提交失败、API TestClient、Gradio 回调和会话加载配置；这些测试不是实际浏览器端验收。

## 13. 跨会话爱好画像

用户画像不从聊天标题或助手总结里直接提取。本阶段只保存兴趣/爱好白名单；姓名、性别、血型、证件号、手机号、地址、积分和献血业务记录仍由入口与业务数据源负责，不进入这个画像模型。

主要调用链：

```text
Gradio 答复交付 / 打开历史 ─┐
FastAPI /chat 响应后后台任务 ─┴→ UserProfileSummarizer.refresh(user_id)
  → conversation_turns 中该 user_id 且超过覆盖游标的用户发言
  → 隐私/业务/静态缓存过滤
  → LLM 结构化候选（hobby、set/revoke/clear、source_turn_id、原话片段）
  → 服务端核对来源轮次、原话包含关系、稳定/一次性/否定条件
  → 单事务写 user_profile_events + user_profile_summaries + extractor_version + 游标
```

`blood_agents/orchestration/user_profile.py` 是提取服务；模型不写数据库。仓储 `save_user_profile_update()` 再核对来源轮次确实属于同一 `user_id/session_id`，并以覆盖游标拒绝迟到/过期的旧结果。事件表只保存来源会话/轮次 ID，不重复保存原始引文；原话如仍在对话保留期内可通过 ID 回查。模型失败时游标不前进，下次刷新可重试；无画像候选时只前进游标，避免重复扫描。

图的 `supervisor` 只把保存的爱好字段加载到 `AgentState.user_profile`。只有检测为用户明确询问个人兴趣/画像时，`ReplyComposer` 才把这份字段加入 general 回复事实；预约需要的献血点/时段偏好仍来自 `long_term_preferences`，且其他领域不接收兴趣字段。新陈述在本轮答复交付后更新，因此当轮回复不会假装已经保存；API 用 FastAPI `BackgroundTasks`，Gradio 用后续页面事件。画像查询进入图前还会重试上次未完成的增量更新；若仍有未处理轮次，`profile_status=pending_update` 会阻止模型把空画像解释为“用户没有爱好”。当前支持模型提取、跨会话读取、纠正/撤销和全清的服务逻辑，没有新增单独的画像管理页面或定时任务。测试使用合成轮次与可控模型，不代表真实供应商模型提取质量或真实用户隐私合规验收。
