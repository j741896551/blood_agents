# 业务模式与动态预约字段

> 本页记录 2026-11-09 的本地集成版本。GitHub `dev` 本次发布图册与文档，运行代码仍使用前一版的 HTTP 和部分 MCP 只读工具。以下完整三模式及动态字段配置用于本地集成版本。

三个模式共用领域流程和业务契约。`BLOOD_AGENT_BUSINESS_TRANSPORT` 决定 Agent 的全部业务调用；模型 API、RAG 检索与会话存储不属于该开关。错误不会跨模式降级。

常用配置已集中在 `.env.example` 的“外部业务接口”一块，只需一个模式字段，其他参数按所选模式生效：

| 模式值 | URL / 文件路径 | 可选鉴权 | 需要启动什么 |
| --- | --- | --- | --- |
| `http` | `BLOOD_AGENT_LEGACY_API_URL` | `BLOOD_AGENT_LEGACY_API_TOKEN` | 业务 HTTP 服务 |
| `mcp` | `BLOOD_AGENT_MCP_URL` | `BLOOD_AGENT_MCP_TOKEN` | MCP 服务 |
| `mock_file`（也接受 `mockfile`） | `BLOOD_AGENT_MOCK_STATE` | 无 | 只启动 Agent |

两个 Token 均作为 `Authorization: Bearer ...` 请求头发送；空值不加此请求头。MCP URL 必须是 MCP Server 的完整端点（常见为 `/mcp`），普通业务 HTTP 地址不能直接作为 MCP 服务。更改配置后重启 Agent。切换模式只切换数据源，不自动迁移地址、订单或其他历史业务数据。

## 不启动业务服务的文件模式

```dotenv
BLOOD_AGENT_BUSINESS_TRANSPORT=mock_file
BLOOD_AGENT_MOCK_STATE=./.local/business.json
# BLOOD_AGENT_MOCK_SEED=./demo_business/seed.json
```

`demo_business/` 是可随仓库分发的独立演示包，不是公司业务实现。JSON 种子包含三个合成账号、实名资料、积分、礼品、历史和献血记录；默认空地址/订单/新预约，可通过对话创建。运行时复制种子后只写 `.local/business.json`，种子不被修改。文件锁与原子替换用于本机演示读写。支持预约/取消/签到/记录、积分/兑换/订单和地址操作；不承诺生产业务规则、分布式锁或大规模文件存储能力。旧的忽略目录 `local_business/` 继续供原有 SQLite/HTTP 测试使用，`local` 仅兼容旧测试。

## HTTP 模式

```dotenv
BLOOD_AGENT_BUSINESS_TRANSPORT=http
BLOOD_AGENT_LEGACY_API_URL=http://127.0.0.1:8001
BLOOD_AGENT_LEGACY_API_TOKEN=
BLOOD_AGENT_LEGACY_API_TIMEOUT=10
# 可选：返回下文 JSON Schema 的接口路径；未配置则兼容原有固定字段。
# BLOOD_AGENT_APPOINTMENT_CONTRACT_PATH=/v1/contracts/appointment
```

全部业务操作通过 `LegacyHTTPClient`。服务端负责数据库和业务校验。HTTP 不会自行推断字段，动态字段需要服务端公布契约。`x-target=root` 可把新增字段加入提交 payload 顶层；默认放在 `payload.extra_fields`。仍保留 `user_id`、确认动作、幂等键等固定边界。

### 本机 8001 与排班排查

配置的业务地址是 `http://127.0.0.1:8001` 时，Agent 调用独立业务进程。Gradio 默认监听 `7861`，进程与业务服务分开启动。本机持有被 Git 忽略的 `local_business/` 时，可用 `.venv312/bin/python -m local_business.http_service` 启动替身，业务库默认是独立的 `local_business.db`。

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
curl --noproxy '*' http://127.0.0.1:8001/health
```

`address already in use` 表示端口已被占用。确认现有进程是业务替身后可直接复用。后台启动时，应查看启动命令指定的日志文件，例如 `tail -f logs/local-business-http.log`。Gradio 的「执行回放」可用 `query_sites` 和 `legacy_http` 核对本轮链路。

本机演示 HTTP 服务启动时读取排班种子，更新种子后需要重启业务进程。2026-11-09 本地检查确认，旧进程仍返回两天，新进程返回未来四天的 12 条排班，其中 4 条余量为零。无排班表示没有对应的排班记录，已约满表示接口返回了记录且余量为零。预约上下文可以影响单轮展示范围，核对全部排班时应先开新对话。

## MCP 模式

连接独立网络 MCP 服务：

```dotenv
BLOOD_AGENT_BUSINESS_TRANSPORT=mcp
BLOOD_AGENT_MCP_URL=http://127.0.0.1:8002/mcp
BLOOD_AGENT_MCP_TIMEOUT=30
# BLOOD_AGENT_MCP_TOKEN=服务端分配的Bearer令牌
```

上述配置默认使用 Streamable HTTP，无需填写第二个传输选择字段。未填写有效 MCP URL 会在构建 adapter 时明确报错，不会自动改用 HTTP 或本地进程。旧配置 `BLOOD_AGENT_MCP_TRANSPORT=http` 兼容为 `streamable_http`。

高级可选：显式使用 stdio，由 Agent 自动启动服务器；服务器可以直接用独立文件数据，不要求 HTTP：

```dotenv
BLOOD_AGENT_BUSINESS_TRANSPORT=mcp
BLOOD_AGENT_MCP_TRANSPORT=stdio
BLOOD_AGENT_MCP_COMMAND=/你的项目路径/.venv312/bin/python
BLOOD_AGENT_MCP_ARGS=["-m","blood_agents.business.mcp_server"]
BLOOD_AGENT_MCP_SERVER_BACKEND=mock_file
BLOOD_AGENT_MOCK_STATE=./.local/business.json
BLOOD_AGENT_MCP_TIMEOUT=30
```

`BLOOD_AGENT_MCP_SERVER_BACKEND` 是内置 MCP **服务端**如何取数据的高级配置，不是 Agent 的第二个业务模式开关。内置 Server 也可设其为 `http` 适配旧 HTTP 服务；这时 Agent 所有业务调用仍只走 MCP，服务器内部再调用 HTTP，HTTP Token 可由 `BLOOD_AGENT_LEGACY_API_TOKEN` 配置。Server 的全部 17 个业务操作按统一 `blood_` 前缀发布，另外提供预约字段契约工具。第一次使用和每次重连时检查所需工具，缺失则失败，不改走 HTTP。查询与写工具分别标注 read-only；注解不替代鉴权。写入仍由领域 Agent 的预览、确认、入口身份和幂等边界控制。

网络客户端代码已实现，目前验证了内存 MCP 协议、本地 stdio 子进程及网络客户端配置/鉴权头构造。真实网络部署、服务端认证与公司业务尚未验收。仓库示例默认 HTTP，已有 `.env` 可参照集中配置块整理；切换模式后须重启 Agent。

## 动态收集新增预约字段

示例：把以下契约放入文件种子的 `appointment_contract`，或由 HTTP 契约接口返回。MCP Server 从其后端取得同一份契约，并嵌入 `blood_submit_appointment` 的 `inputSchema.properties.payload.properties.extra_fields`。MCP 客户端从 `tools/list` 读取该 schema，HTTP 客户端从配置的契约路径读取。

```json
{
  "version": "2",
  "type": "object",
  "properties": {
    "gender": {"type": "string", "title": "本次献血人性别", "enum": ["男", "女"], "x-source": "profile.gender"},
    "contact_preference": {"type": "string", "title": "通知方式", "enum": ["短信", "电话"]},
    "remark": {"type": "string", "title": "备注", "maxLength": 100}
  },
  "required": ["gender", "contact_preference"],
  "additionalProperties": false
}
```

预约理解模型接收此 schema，只把用户明确提供的值提取到 `slots.extra_fields`。程序用 JSON Schema 校验后增量合并；本人资料中的性别/血型可按显式 `x-source` 自动补齐，代他人仍需收集。缺必填项时给回复模型缺项与字段说明；可选项不阻塞预约。齐全后预览包含新增字段，下一轮明确确认才提交。契约指纹发生变化会撤销旧确认，必须重新预览；可恢复的缺字段接口拒绝保留已收集值。

首版支持最多 30 个普通 string/integer/number/boolean 字段、枚举、长度和数值范围。`user_id` 等身份/状态/权限字段不能动态开放给模型。嵌套复杂条件、任意 `$ref`、新增业务动作、未知资料来源与任意第三方工具返回格式不在自动适配范围内；不能仅修改说明文字来绕过契约校验。MCP Server 公布的工具 schema 在启动时生成，修改其后端契约后需重启 Server；客户端每轮重新发现 schema。生产服务应提供稳定和版本化的工具契约。

本机可选网络 Server 启动方式（另一个终端）：

```bash
BLOOD_AGENT_MCP_SERVER_BACKEND=mock_file \
BLOOD_AGENT_MCP_SERVER_TRANSPORT=streamable_http \
BLOOD_AGENT_MCP_SERVER_PORT=8002 \
.venv312/bin/python -m blood_agents.business.mcp_server
```

内置网络演示入口仅绑定 `127.0.0.1`，未配置生产服务端认证。对接有认证的外部 MCP 可在 Agent 客户端配置 Bearer 令牌。克隆后按 README 安装依赖，设 `mock_file` 再启动网页即可使用合成数据；自然对话仍需要配置可用的模型 API。
