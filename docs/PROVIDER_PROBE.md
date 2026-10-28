# 第三方模型平台只读探测

本工具用于补足本地 HTTP Mock 无法证明的供应商连通性与小规模并发证据。它**不走业务图、不读取用户/预约数据、不调用写工具**；每次只发送固定合成话术。默认只打印计划，必须显式加 `--execute` 才会发生收费的外部 API 请求。单次只选一个供应商，最多并发 4 条请求，每条最多请求 128 个输出 token；SDK 自动重试沿用项目配置关闭。

先从平台控制台核对账户余额、免费额度/价格和实际限流，再逐级执行，不要把文档公布的上限当作当前账号的承诺。当前配置使用的模型可参考 [DeepSeek 模型价格](https://api-docs.deepseek.com/quick_start/pricing/)、[DeepSeek 限流](https://api-docs.deepseek.com/quick_start/rate_limit/)、[百炼 qwen3.8-flash 模型价格](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)、[百炼限流](https://help.aliyun.com/zh/model-studio/rate-limit) 和 [百炼账号模型限额查询](https://help.aliyun.com/zh/model-studio/list-quotas)。价格/额度会变化，以调用当日的官方页面及账号控制台为准。不要在报告里写 API Key、完整请求/回复或真实用户资料。

```bash
# 只打印模型、超时和预计请求数；不发请求
.venv312/bin/python -m blood_agents.evals.provider_probe --provider primary --parallel 1
.venv312/bin/python -m blood_agents.evals.provider_probe --provider backup --parallel 1

# 核对费用边界后才显式执行；从 1 开始，再考虑 2、4
.venv312/bin/python -m blood_agents.evals.provider_probe --provider primary --parallel 1 --execute
.venv312/bin/python -m blood_agents.evals.provider_probe --provider backup --parallel 1 --execute
.venv312/bin/python -m blood_agents.evals.provider_probe --provider primary --parallel 2 --execute
```

结果只记录每条请求的成功/失败、异常类别、HTTP 状态、耗时、可见输出长度和供应商若返回的 token 用量，不记录回复正文。`p50/p95` 仅为本次极小样本的最近秩值，不代表平台 SLA。先分别验证两个服务的独立连接，再按 1→2→4 做只读阶梯；如果出现 429、超时或 `empty_response`，停止升级并检查账号限额、网络和模型配置。模型输出为空不等于网络失败，也可能是 128-token 上限下没有产生最终可见文本。

这个入口**不验证 LangGraph 主备自动切换**。主备接线和 FastAPI `/chat` 的本地 HTTP Mock 故障测试已存在；真实备用服务连通后，仍需单独验证真实备用接续导航/回复，并核对没有重复业务写入。应用默认每供应商在途上限为 1；本探测为测供应商容量临时使用 2/4 路并行，不修改应用配置或证明当前 `/chat` 能达到相同吞吐。将来调高应用在途配置和缩短 40/20/60 秒开发期预算，应依据重复测量、费用和失败样本另行决定。
