# 项目开发约定

## 项目目标与技术背景

本项目用于展示简历中的“血站智慧献血导航”多智能体技术实践。业务场景作为智能体交互与工具编排的演示载体；预约库存、积分结算等旧业务系统内部规则不作为开发主线。

- 当前技术栈：Python 3.10+、LangChain、LangGraph、FastAPI、Gradio、SQLite、SQLAlchemy 2.x（Agent 自有表）；本阶段 MCP 在 Python 3.12 隔离环境验证。
- 当前计划覆盖的技术方向：平台大模型 API 的主备调用与容灾、MCP、bge-large-zh-v1.5、bge-reranker、FAISS、Milvus。用户已决定暂缓 OCR 与 LoRA 微调，并明确本项目不做本地大模型部署；`deploy/` 中 vLLM/TGI 仅为未实跑的历史模板，不是当前任务或已实现能力。待用户重新确认后才恢复这些范围。
- 目录：`blood_agents/orchestration/` 主图与导航；`blood_agents/domains/` 各领域 Agent；`blood_agents/llm/` 模型访问与回复；`blood_agents/business/` 外部业务契约、HTTP 客户端、MCP 服务和工具；`blood_agents/storage/` Agent 会话与偏好仓储；`blood_agents/knowledge/` 检索；`blood_agents/core/` 共享状态契约；`blood_agents/log/` 追踪；`blood_agents/ui/` 展示辅助；`blood_agents/evals/` 评测器。被 Git 忽略的同级 `local_business/` 仅供本机替代旧系统；发布 Agent 代码时不包含。`tests/` 测试；`docs/` 代码说明；顶层 `evals/` 用例。
- 当前范围、任务和验收标准以 `PROJECT.md` 为唯一管理索引。保留 `README.md` 和 `docs/CODE_WALKTHROUGH.md` 的使用说明与解读用途。

## 开发规则

1. 开发前读取 `PROJECT.md`；保留现有架构和已完成代码，不重新规划已经完成的功能。
2. 优先推进模型理解、智能体协作、状态流转、工具调用及可观察性。外部业务替身返回满足本机测试需要的结果即可，不扩展旧业务系统内部实现。
3. 新功能、技术替换和范围扩展需要用户确认。当前已确认的技术覆盖可按开发清单推进。
4. 用户身份来自入口上下文；会话应隔离；预约等提交动作须经用户确认。工具返回值决定实际操作结果。
5. 依赖按需加载，保持无 API Key 的基础演示可运行。真实模型与服务验证须记录所用环境，不能用 Mock 结果代替。
6. 记录状态时区分已实现、已验证、待实现、未验证；不以文件存在等同端到端完成。
7. 简历中的生产指标不能作为本仓库的实测指标。评测报告必须说明样本、方法、环境与局限。
8. 核心流程保留有助于理解设计的中文注释。个人管理偏好不写入公开文件；本地私有文件按需创建并检查 Git 忽略状态。
9. 未经授权，不自动 Commit、Push、发布，不操作真实生产业务或删除数据。
10. 业务 Agent 的用户可见结论须以本轮已查询的业务数据或已执行工具回执为依据。无数据、查询失败、功能仅占位时，不能由模型补造记录、状态或操作成功；“退出未提交预约意向”不得称作“取消已提交预约”。
11. 献血记录只按入口登录 `user_id` 查询外部业务接口；公开资料只用于设计字段，不作为个人献血事实来源。全血毫升数与单采治疗量分开统计，未知字段保持未知。
12. 当前 Agent 会话/偏好和独立文件中的 LangGraph checkpoint 均用 SQLAlchemy 模型读写 SQLite；checkpoint 保持旧 `SqliteSaver` 的表结构与序列化格式。MySQL、Alembic、Redis 和多 Worker 一致性仍未实现，不能表述为已完成。

## 启动与验证

```bash
# 首次安装基础服务、网页、MCP 与开发依赖（用 Python 3.10+ 创建环境）
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -e '.[dev,api,ui,mcp]'

# 网页演示
GRADIO_SERVER_PORT=7861 .venv312/bin/python -m blood_agents.web

# API / CLI
.venv312/bin/python -m uvicorn blood_agents.api:app --reload
.venv312/bin/python -m blood_agents.cli

# 回归验证（缓存放在可写临时目录）
PYTHONPYCACHEPREFIX=/private/tmp/blood-agent-pycache .venv312/bin/python -m pytest -q
.venv312/bin/ruff check .
.venv312/bin/python -m blood_agents.evals.evaluation
```

仅修改管理文档时检查内容和 Git 状态即可；修改功能时运行对应验证。重组件的依赖、模型下载和部署验证按当前任务需要执行。
