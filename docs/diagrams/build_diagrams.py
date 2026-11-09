"""从同一份声明生成可编辑图、Mermaid、SVG、PNG和来源索引；不启动应用或访问模型。"""

from __future__ import annotations

import argparse
import ast
import html
import itertools
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from sequence_sources import SEQUENCES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAPSHOT_DATE = "2026-11-09"
SCOPE = "本地图谱快照。GitHub dev 此次仅发布文档，部分功能仍在本地集成。"
PALETTE = {
    "entry": ("#F1F5F9", "#64748B"),
    "orchestrator": ("#EAF2FF", "#3575C5"),
    "agent": ("#E9F6F1", "#288366"),
    "tool": ("#EFF8EC", "#66933B"),
    "storage": ("#F2EDFC", "#8867B4"),
    "model": ("#F4EEFF", "#8C60BD"),
    "decision": ("#FFF4DE", "#BD8833"),
    "error": ("#FFF0EE", "#CA6B5A"),
    "planned": ("#F5F5F5", "#87909D"),
    "external": ("#FFF1E5", "#BB7D42"),
}


def node(key, label, col, row, kind="agent"):
    """声明语义节点及网格位置；文字和几何由各格式共用。"""
    return {"id": key, "label": label, "col": col, "row": row, "kind": kind}


def edge(source, target, label="", kind="control"):
    """声明有代码依据的调用/状态连线；规划视图单独标记。"""
    return {"source": source, "target": target, "label": label, "kind": kind}


def view(key, title, subtitle, nodes, edges, refs, guide, planned=False):
    """创建一页图及阅读说明，避免把规划内容混入当前实现。"""
    return {
        "id": key,
        "title": title,
        "subtitle": subtitle.replace("现有实现", "本地集成", 1),
        "nodes": nodes,
        "edges": edges,
        "refs": refs,
        "guide": guide,
        "planned": planned,
    }


VIEWS = [
    view(
        "01-system-context",
        "系统全景",
        "现有实现｜模型理解与表达，程序编排与检查，业务后端决定事实",
        [
            node("User", "用户\n交互与最终确认", 0, 1, "entry"),
            node(
                "ConversationGraph",
                "智慧献血 Agent 服务\nLangChain / LangGraph\nGradio / FastAPI / CLI",
                1,
                1,
                "orchestrator",
            ),
            node("ModelGateway", "平台大模型 API\n主备 / 预算 / 熔断", 2, 0, "model"),
            node("LegacyBloodAPI", "外部业务能力\nHTTP / MCP / mock_file", 2, 1, "external"),
            node("KnowledgeWorkspace", "知识工作区与检索\n发布资料 / BGE / 索引", 2, 2, "tool"),
            node("AgentRepository", "Agent 自有持久化\n会话 / 偏好 / checkpoint", 1, 3, "storage"),
        ],
        [
            edge("User", "ConversationGraph", "用户消息"),
            edge("ConversationGraph", "ModelGateway", "语义理解 / 回复"),
            edge("ConversationGraph", "LegacyBloodAPI", "业务工具"),
            edge("ConversationGraph", "KnowledgeWorkspace", "知识查询 / 管理"),
            edge("ConversationGraph", "AgentRepository", "保存与恢复", "state"),
        ],
        [
            ("blood_agents/orchestration/conversation.py", "class ConversationGraph"),
            ("blood_agents/business/transport.py", "def business_api_from_env"),
        ],
        "这是系统边界图。业务数据不属于 Agent 数据库；大模型由平台 API 提供，本地 BGE 只承担编码与精排。",
    ),
    view(
        "02-orchestration",
        "主导航与多智能体编排",
        "现有实现｜四个活动业务领域＋一个暂缓的 OCR 注册项；多任务顺序执行",
        [
            node(
                "ConversationGraph_invoke",
                "ConversationGraph.invoke\n用户 / 会话隔离与锁",
                1,
                0,
                "entry",
            ),
            node("static_faq_cache", "static_faq_cache\n完整唯一的审核缓存", 1, 1, "decision"),
            node("supervisor", "supervisor\n模型导航 / 账户范围", 1, 2, "orchestrator"),
            node("AgentRegistry", "AgentRegistry\n目录 / schema / 接线", 0, 2, "storage"),
            node("main_agent_response", "general_reply\n问候 / 范围拒绝", 2, 2, "agent"),
            node("appointment_subgraph", "appointment_subgraph\n预约 / 查询 / 取消 / 签到", 0, 3),
            node("points_subgraph", "points_subgraph\n爱心积分 / 订单 / 地址", 1, 3),
            node("records_subgraph", "records_subgraph\n个人献血记录查询", 2, 3),
            node("faq_subgraph", "faq_subgraph\n知识与业务组合只读问答", 0, 4),
            node("ocr_subgraph", "ocr_subgraph\n原型注册保留 / 暂缓", 2, 4, "planned"),
            node("TaskCoordinator", "TaskCoordinator\n顺序派发 / 等待时暂停", 1, 4, "orchestrator"),
            node("main_agent_return", "agent_handoff\n等待用户或回交主导航", 1, 5, "decision"),
            node("END", "本轮结束\ncheckpoint 续接下一轮", 1, 6, "entry"),
        ],
        [
            edge("ConversationGraph_invoke", "static_faq_cache"),
            edge("static_faq_cache", "END", "唯一命中"),
            edge("static_faq_cache", "supervisor", "未命中"),
            edge("AgentRegistry", "supervisor", "动态能力目录", "state"),
            edge("supervisor", "main_agent_response", "general"),
            edge("supervisor", "appointment_subgraph", "appointment"),
            edge("supervisor", "points_subgraph", "points"),
            edge("supervisor", "records_subgraph", "records"),
            edge("supervisor", "faq_subgraph", "faq"),
            edge("supervisor", "TaskCoordinator", "多领域任务"),
            edge("supervisor", "ocr_subgraph", "暂缓注册项", "planned"),
            *[
                edge(x, "main_agent_return", "交接结果")
                for x in [
                    "appointment_subgraph",
                    "points_subgraph",
                    "records_subgraph",
                    "faq_subgraph",
                ]
            ],
            edge("TaskCoordinator", "main_agent_return", "逐项执行后交接"),
            edge("main_agent_response", "END"),
            edge("main_agent_return", "END"),
        ],
        [
            ("blood_agents/orchestration/graph.py", "builder = StateGraph(AgentState)"),
            ("blood_agents/orchestration/agent_registry.py", "class AgentRegistry"),
        ],
        "每轮都会经过主导航；活动业务通过 checkpoint 续接。领域子图不是并行 fan-out，主图也没有独立 Skill 加载器。",
    ),
    view(
        "03-appointment",
        "预约全流程",
        "现有实现｜局部网点澄清是专用分支；统一临时确认队列仍待实现",
        [
            node(
                "appointment_understand",
                "appointment_understand\n模型提取 action / slots",
                1,
                0,
                "orchestrator",
            ),
            node("AppointmentToolPlanner", "AppointmentToolPlanner\n只读工具规划", 1, 1, "model"),
            node("query_sites", "资料 / 历史 / 号源查询\n可信入口 user_id", 1, 2, "tool"),
            node(
                "_site_clarification", "_site_clarification\n未知网点 / 候选澄清", 0, 3, "decision"
            ),
            node("missing", "检查必填与参数约束\n不足则 collecting", 1, 3, "decision"),
            node("query_actions", "query / cancel / checkin\n查询或单独确认已有记录", 2, 2, "tool"),
            node("awaiting_confirmation", "awaiting_confirmation\n生成完整预约预览", 1, 4, "agent"),
            node("preview_delivered", "preview_delivered\n可交付回复后允许确认", 1, 5, "decision"),
            node("submit_appointment", "submit_appointment\n重查号源 / 幂等提交", 1, 6, "tool"),
            node("appointment_reply", "appointment_reply\n根据接口回执模型表达", 2, 6, "model"),
            node(
                "appointment_handoff",
                "appointment_handoff\n等待或清草稿并回交",
                2,
                7,
                "orchestrator",
            ),
            node("User_reply", "用户补充 / 修改 / 澄清\n下一轮重新理解", 0, 5, "entry"),
        ],
        [
            edge("appointment_understand", "AppointmentToolPlanner"),
            edge("AppointmentToolPlanner", "query_sites"),
            edge("appointment_understand", "query_actions", "其他预约操作"),
            edge("query_sites", "missing"),
            edge("query_sites", "_site_clarification", "名称未匹配"),
            edge("_site_clarification", "User_reply", "局部选择"),
            edge("missing", "User_reply", "缺项 / 无可用组合"),
            edge("User_reply", "appointment_understand", "下一轮"),
            edge("missing", "awaiting_confirmation", "资料完整且组合可约"),
            edge("awaiting_confirmation", "preview_delivered"),
            edge("preview_delivered", "submit_appointment", "后续明确最终确认"),
            edge("submit_appointment", "appointment_reply", "成功 / 拒绝"),
            edge("query_actions", "appointment_reply", "查询结果 / 操作回执"),
            edge("appointment_reply", "appointment_handoff"),
        ],
        [
            (
                "blood_agents/domains/appointment.py",
                'if (slots.get("status") == "awaiting_confirmation" and slots.get("preview_delivered")',
            ),
            ("blood_agents/orchestration/graph.py", "def appointment_handoff"),
        ],
        "收集无固定字段顺序，任何修改会撤销旧提交预览；局部确认只选网点。取消、签到有各自的待确认目标和回执，不能与退出未提交意向混称。",
    ),
    view(
        "04-points",
        "爱心积分与纪念品",
        "现有实现｜读请求直接查询；兑换、地址保存、取消订单均先预览后确认",
        [
            node(
                "points_understand", "points_understand\n模型理解动作与参数", 1, 0, "orchestrator"
            ),
            node("points_query", "query / catalog / recommend\n余额与纪念品图文目录", 0, 1, "tool"),
            node("points_orders", "orders / order_status\n订单与物流状态", 2, 1, "tool"),
            node("redeem", "redeem\n核对候选 / 积分 / 库存", 1, 1, "decision"),
            node("reward_selection", "reward_selection\n近似名称候选由用户选择", 0, 2, "decision"),
            node("shipping", "收货地址查询与选择\n缺失时先创建地址", 1, 2, "tool"),
            node("address_preview", "address_add / update\n地址预览与独立确认", 1, 3, "decision"),
            node(
                "save_shipping_address", "save_shipping_address\n成功后续接兑换预览", 1, 4, "tool"
            ),
            node(
                "order_preview",
                "awaiting_redemption_confirmation\n纪念品 / 积分 / 收货信息",
                1,
                5,
                "decision",
            ),
            node("redeem_reward", "redeem_reward\n最终确认才请求下单", 1, 6, "tool"),
            node(
                "order_actions", "取消订单 / 修改收货信息\n默认地址等操作单独确认", 2, 3, "decision"
            ),
            node("points_reply", "points_reply\n模型依据工具事实回复", 2, 6, "model"),
            node("points_handoff", "points_handoff\n保留待办或完成回交", 2, 7, "orchestrator"),
        ],
        [
            edge("points_understand", "points_query", "只读"),
            edge("points_understand", "points_orders", "只读"),
            edge("points_understand", "redeem", "兑换意向"),
            edge("redeem", "reward_selection", "名称需澄清"),
            edge("reward_selection", "shipping", "明确选中"),
            edge("redeem", "shipping", "礼品已明确且足够"),
            edge("shipping", "address_preview", "没有可用地址"),
            edge("address_preview", "save_shipping_address", "确认保存"),
            edge("save_shipping_address", "order_preview", "保存回执成功"),
            edge("shipping", "order_preview", "已明确选择已有地址"),
            edge("order_preview", "redeem_reward", "最终确认兑换"),
            edge("redeem_reward", "points_reply", "下单回执决定结果"),
            edge("points_query", "points_reply"),
            edge("points_orders", "points_reply"),
            edge("points_understand", "order_actions"),
            edge("order_actions", "points_reply", "确认后接口回执"),
            edge("points_reply", "points_handoff"),
        ],
        [
            ("blood_agents/domains/points.py", "def points_agent"),
            ("blood_agents/llm/points_query_reply.py", "class PointsQueryText"),
        ],
        "地址保存确认与兑换下单确认是两个动作。当前只传接口已有的介绍、规格与图片，不把描述性型号猜成可下单 SKU。",
    ),
    view(
        "05-records-faq",
        "记录查询与组合只读问答",
        "现有实现｜模型有界选择工具、读取观察、再决定下一步；使用结构化 JSON",
        [
            node("supervisor", "supervisor\n单独记录或组合分析", 1, 0, "orchestrator"),
            node("records_agent", "records_agent\n只查入口用户献血记录", 0, 1, "agent"),
            node("ReadOnlyQueryAgent", "ReadOnlyQueryAgent.run\nFAQ 的模型决策循环", 1, 1, "model"),
            node(
                "consultation_catalog", "consultation_catalog\n动态只读工具白名单", 2, 1, "storage"
            ),
            node("QueryStep", "QueryStep\n工具 / 参数 / 观察引用", 1, 2, "decision"),
            node(
                "invoke_consultation", "invoke_consultation\nschema 校验 / 注入身份", 1, 3, "tool"
            ),
            node("knowledge_search", "knowledge_search\n已发布知识片段", 0, 4, "tool"),
            node("query_records", "个人查询工具\n记录 / 积分 / 订单等", 1, 4, "tool"),
            node("date_offset", "date_offset\n引用事实与规则进行日期运算", 2, 4, "tool"),
            node("EvidenceAssessor", "EvidenceAssessor\n知识证据核验 / 缺口", 0, 5, "decision"),
            node("observations", "observations\n成功 / 失败及来源观察", 1, 5, "storage"),
            node("ReplyComposer", "ReplyComposer\n事实与证据辅助模型回复", 1, 6, "model"),
        ],
        [
            edge("supervisor", "records_agent", "单独记录"),
            edge("supervisor", "ReadOnlyQueryAgent", "知识 / 组合只读问题"),
            edge("consultation_catalog", "ReadOnlyQueryAgent", "工具目录", "state"),
            edge("ReadOnlyQueryAgent", "QueryStep"),
            edge("QueryStep", "invoke_consultation", "call_tool"),
            edge("invoke_consultation", "knowledge_search"),
            edge("invoke_consultation", "query_records"),
            edge("QueryStep", "date_offset", "内置运算"),
            edge("knowledge_search", "EvidenceAssessor"),
            edge("EvidenceAssessor", "observations"),
            edge("query_records", "observations"),
            edge("date_offset", "observations"),
            edge("observations", "ReadOnlyQueryAgent", "再决策；默认上限8步", "state"),
            edge("QueryStep", "ReplyComposer", "finish"),
            edge("records_agent", "ReplyComposer", "记录回执"),
        ],
        [
            ("blood_agents/llm/query_agent.py", "class ReadOnlyQueryAgent"),
            ("blood_agents/domains/records.py", "def records_agent"),
        ],
        "个人下次献血参考日期不走固定关键词工作流；模型自行决定查询记录和规则，再引用来源做运算。无充分证据不能生成医学资格结论。",
    ),
    view(
        "06-rag-pipeline",
        "RAG 知识库生命周期",
        "现有实现｜草稿测试与正式索引分离；没有预置发布语料时按空库处理",
        [
            node("rag_management", "Gradio 知识库管理\n上传 / 清洗 / 切分 / 发布", 1, 0, "entry"),
            node(
                "KnowledgeWorkspace",
                "KnowledgeWorkspace\n提取完整 PDF / Markdown / TXT",
                1,
                1,
                "storage",
            ),
            node("confirm_cleaning", "核对原件与提取结果\n确认清洗后的正文", 1, 2, "decision"),
            node(
                "DocumentChunkingPipeline",
                "DocumentChunkingPipeline\n章节 / 段落 / 滑窗 / 语义",
                1,
                3,
                "tool",
            ),
            node("temporary_retriever", "临时 BGE 召回\n单题测试 / 策略对比", 0, 4, "tool"),
            node(
                "confirm_evaluation",
                "confirm_chunking_evaluation\n人工确认当前参数合格",
                1,
                4,
                "decision",
            ),
            node("publish", "publish + refresh\n构建成功才替换活动检索器", 1, 5, "orchestrator"),
            node("default_retriever", "default_retriever\n当前已发布文档集合", 1, 6, "storage"),
            node("BGEEncoder", "BGEEncoder\nbge-large-zh-v1.5", 0, 6, "model"),
            node(
                "vector_store",
                "FAISSVectorStore\n或 MilvusVectorStore\n向量召回 / 绝对阈值",
                0,
                7,
                "storage",
            ),
            node("HybridRetriever", "HybridRetriever.search\n词面分数＋向量分数相加", 1, 7, "tool"),
            node("BGEReranker", "BGEReranker（可选）\n只排序已过门槛的候选", 1, 8, "model"),
            node(
                "knowledge_evidence",
                "文本 / 来源 / 元数据\n正式索引检索 / FAQ 证据核验",
                1,
                9,
                "entry",
            ),
        ],
        [
            edge("rag_management", "KnowledgeWorkspace"),
            edge("KnowledgeWorkspace", "confirm_cleaning", "待确认草稿"),
            edge("confirm_cleaning", "DocumentChunkingPipeline", "确认正文"),
            edge("DocumentChunkingPipeline", "temporary_retriever", "发布前测试"),
            edge("temporary_retriever", "confirm_evaluation", "人工检查"),
            edge("confirm_evaluation", "publish"),
            edge("publish", "default_retriever"),
            edge("BGEEncoder", "vector_store"),
            edge("default_retriever", "HybridRetriever"),
            edge("default_retriever", "BGEEncoder", "启用向量后端时"),
            edge("vector_store", "HybridRetriever"),
            edge("HybridRetriever", "BGEReranker", "启用精排时"),
            edge("HybridRetriever", "knowledge_evidence", "未启用精排"),
            edge("BGEReranker", "knowledge_evidence"),
        ],
        [
            ("blood_agents/knowledge/rag.py", "def search(self, query"),
            ("blood_agents/knowledge/workspace.py", "def confirm_cleaning"),
            ("blood_agents/knowledge/workspace.py", "def publish"),
        ],
        "memory 是内存词面检索，不是向量数据库，也不是长期记忆。当前融合不是 BM25 或 RRF；草稿不进入聊天，文档冲突没有自动版本优先级硬过滤。",
    ),
    view(
        "07-business-transports",
        "业务接口与三种后端模式",
        "现有实现｜一个模式覆盖全部业务读写，错误不跨模式回退",
        [
            node(
                "ToolRegistry",
                "ToolRegistry.invoke\n身份绑定 / 追踪 / 工具边界",
                1,
                0,
                "orchestrator",
            ),
            node(
                "business_api_from_env",
                "business_api_from_env\n按业务模式配置选择后端",
                1,
                1,
                "decision",
            ),
            node("LegacyHTTPClient", "http：LegacyHTTPClient\nURL / Bearer Token", 0, 2, "tool"),
            node(
                "MCPBusinessQueryClient",
                "mcp：MCPBusinessQueryClient\nStreamable HTTP / stdio",
                1,
                2,
                "tool",
            ),
            node(
                "FileBusinessAPI", "mock_file：FileBusinessAPI\n独立 demo_business 包", 2, 2, "tool"
            ),
            node(
                "HTTP_business", "独立业务 HTTP 服务\n本机示例监听 127.0.0.1:8001", 0, 4, "external"
            ),
            node("business_sqlite", "本机 HTTP 演示业务库\n独立 SQLite", 0, 5, "storage"),
            node("build_mcp_server", "build_mcp_server\n协议工具＋动态预约契约", 1, 3, "external"),
            node("MCP_backend", "MCP 服务端后端\nHTTP 适配器或文件实现", 1, 4, "decision"),
            node("business_json", "独立 JSON 业务数据\n.local/business.json", 2, 4, "storage"),
        ],
        [
            edge("ToolRegistry", "business_api_from_env"),
            edge("business_api_from_env", "LegacyHTTPClient", "http"),
            edge("business_api_from_env", "MCPBusinessQueryClient", "mcp"),
            edge("business_api_from_env", "FileBusinessAPI", "mock_file"),
            edge("LegacyHTTPClient", "HTTP_business"),
            edge("HTTP_business", "business_sqlite", "读写业务事实"),
            edge("MCPBusinessQueryClient", "build_mcp_server"),
            edge("build_mcp_server", "MCP_backend"),
            edge("MCP_backend", "HTTP_business", "选择 http"),
            edge("MCP_backend", "business_json", "选择 mock_file"),
            edge("FileBusinessAPI", "business_json"),
        ],
        [
            ("blood_agents/business/transport.py", "def business_api_from_env"),
            ("blood_agents/business/mcp_server.py", "for name in OPERATIONS"),
        ],
        "HTTP 演示由独立进程监听 8001，和 Gradio 分开。MCP 服务端可选文件后端；切换模式不迁移历史地址或订单。local 仅为旧测试兼容。",
    ),
    view(
        "08-memory-cache",
        "会话、记忆与缓存",
        "现有实现｜业务数据、会话日志、流程快照和标准问答缓存分别管理",
        [
            node(
                "ConversationGraph",
                "ConversationGraph.invoke\n按 user_id + session_id 加锁",
                1,
                0,
                "orchestrator",
            ),
            node(
                "SQLAlchemySqliteSaver",
                "SQLAlchemySqliteSaver\n独立 checkpoint 文件",
                0,
                1,
                "storage",
            ),
            node("AgentRepository", "AgentRepository\nAgent 自有 SQLite 六类表", 1, 2, "storage"),
            node(
                "project_short_term_memory",
                "project_short_term_memory\n从流程状态投影任务快照",
                0,
                3,
                "tool",
            ),
            node(
                "LongTermMemoryService", "LongTermMemoryService\n明确网点 / 时段偏好", 1, 3, "model"
            ),
            node("UserProfileSummarizer", "UserProfileSummarizer\n稳定兴趣增量归纳", 2, 3, "model"),
            node("ConversationSummarizer", "ConversationSummarizer\n会话标题与摘要", 1, 4, "model"),
            node("privacy", "隐私 / 原话证据 / 来源门禁\n身份资料不进入兴趣画像", 2, 4, "decision"),
            node("StaticFaqCache", "StaticFaqCache\n审核 / 版本 / 有效期", 0, 5, "storage"),
            node("next_turn", "下一轮输入\n按需读取上下文与偏好", 1, 6, "entry"),
        ],
        [
            edge("ConversationGraph", "SQLAlchemySqliteSaver", "每轮恢复与保存", "state"),
            edge("ConversationGraph", "AgentRepository", "脱敏对话与元数据", "state"),
            edge("SQLAlchemySqliteSaver", "project_short_term_memory", "恢复流程后投影", "state"),
            edge("AgentRepository", "LongTermMemoryService"),
            edge("AgentRepository", "UserProfileSummarizer"),
            edge("AgentRepository", "ConversationSummarizer"),
            edge("UserProfileSummarizer", "privacy"),
            edge("project_short_term_memory", "next_turn"),
            edge("LongTermMemoryService", "next_turn"),
            edge("privacy", "next_turn"),
            edge("StaticFaqCache", "next_turn", "完整唯一命中可直接回答"),
        ],
        [
            ("blood_agents/orchestration/conversation.py", "def invoke(self, state"),
            ("blood_agents/storage/models.py", "class ConversationTurn"),
        ],
        "标题摘要不是预约授权；缓存仅适用于无个人/实时业务依赖的标准问答。会话表与 checkpoint 为不同 SQLite 文件，保留期清理不会自动删除所有存储。",
    ),
    view(
        "09-failover-concurrency",
        "模型容灾、并发与回复边界",
        "现有实现｜同会话串行、跨会话可并行；不是领域子智能体并行执行",
        [
            node("entry_queue", "Gradio queue / FastAPI\n入口并发与会话锁", 1, 0, "entry"),
            node(
                "ModelGateway_call",
                "ModelGateway.call / stream\n每次调用共享预算",
                1,
                1,
                "orchestrator",
            ),
            node(
                "ModelProvider_primary", "主服务 ModelProvider\n在途限制 / 熔断状态", 1, 2, "model"
            ),
            node("validate", "结构化输出 / 回复校验\n事实与隐私边界", 1, 3, "decision"),
            node(
                "ModelProvider_backup", "备用 ModelProvider\n独立 URL / Key / 额度", 2, 3, "model"
            ),
            node("ReplyComposer", "ReplyComposer\n必要时表达重写一次", 1, 4, "model"),
            node("reply_delivery", "回复交付\n业务 / FAQ 完整后显示", 1, 5, "entry"),
            node("unavailable", "模型服务不可用\n明确错误，不生成业务结论", 2, 5, "error"),
            node("tool_receipt", "业务工具回执\n写结果不随回复失败重试", 0, 4, "storage"),
            node("risk_agent", "RiskAgent 内容风控\n默认 passthrough / 暂缓完善", 0, 6, "planned"),
        ],
        [
            edge("entry_queue", "ModelGateway_call"),
            edge("ModelGateway_call", "ModelProvider_primary"),
            edge("ModelProvider_primary", "validate", "返回结果"),
            edge(
                "ModelProvider_primary", "ModelProvider_backup", "超时 / 错误 / 熔断 / 忙", "error"
            ),
            edge("ModelProvider_backup", "validate", "备用返回"),
            edge("validate", "ReplyComposer", "可用事实"),
            edge("ReplyComposer", "reply_delivery", "通过关键检查"),
            edge("ModelProvider_backup", "unavailable", "仍不可用", "error"),
            edge("ReplyComposer", "unavailable", "校验或改写仍失败", "error"),
            edge("tool_receipt", "ReplyComposer", "回执事实", "state"),
            edge("risk_agent", "reply_delivery", "仅显式启用后审查", "planned"),
        ],
        [
            ("blood_agents/llm/model_gateway.py", "def call(self, messages"),
            ("blood_agents/llm/reply_composer.py", "can_stream = ("),
        ],
        "默认主40秒、备20秒、单次预算60秒，不是整轮对话 SLA。超时后台请求可能稍后结束；普通 general 的保守流式不代表业务回答支持分段风控。",
    ),
    view(
        "11-state-dataflow",
        "结构化状态与数据流",
        "现有实现｜模型输出动作和槽位，程序保存流程状态并执行工具",
        [
            node("HumanMessage", "HumanMessage\n用户本轮原话", 0, 0, "entry"),
            node("AgentState", "AgentState\nmessages / active_agent / results", 1, 1, "storage"),
            node(
                "NavigationDecision",
                "NavigationDecision\n意图 / 账户范围 / 任务片段",
                0,
                2,
                "model",
            ),
            node(
                "AppointmentUnderstanding",
                "AppointmentUnderstanding\naction / slots / site_candidates",
                1,
                2,
                "model",
            ),
            node("PointsUnderstanding", "PointsUnderstanding\n动作与明确业务字段", 2, 2, "model"),
            node(
                "domain_validation",
                "领域执行器与字段契约\n格式 / 来源 / 版本 / 确认",
                1,
                3,
                "decision",
            ),
            node("ToolRegistry", "ToolRegistry\n入口身份绑定与调用", 1, 4, "tool"),
            node("result", "本轮 result\n业务回执 / 检索观察", 1, 5, "storage"),
            node("ReplyComposer", "ReplyComposer\n模型表达＋后置检查", 1, 6, "model"),
            node("AIMessage", "AIMessage\n用户答复与来源状态", 1, 7, "entry"),
        ],
        [
            edge("HumanMessage", "AgentState"),
            edge("AgentState", "NavigationDecision"),
            edge("AgentState", "AppointmentUnderstanding"),
            edge("AgentState", "PointsUnderstanding"),
            edge("NavigationDecision", "domain_validation"),
            edge("AppointmentUnderstanding", "domain_validation"),
            edge("PointsUnderstanding", "domain_validation"),
            edge("domain_validation", "ToolRegistry"),
            edge("ToolRegistry", "result"),
            edge("result", "ReplyComposer"),
            edge("ReplyComposer", "AIMessage"),
            edge("AIMessage", "AgentState", "add_messages reducer", "state"),
        ],
        [
            ("blood_agents/core/models.py", "class AgentState"),
            ("blood_agents/orchestration/router.py", "def interpret_appointment"),
        ],
        "action=confirm 是模型动作，不等于预约已经成功；agent_status=completed 表示流程结束，也可能是退出或失败。当前仍混有固定词和正则校正。",
    ),
    view(
        "12-planned-confirmation",
        "待实现：统一临时确认机制",
        "待实现｜任务3l / 3m，不能作为当前已经具备的能力",
        [
            node("Collecting", "正常收集有效资料", 1, 0, "planned"),
            node(
                "NeedsClarification", "仅歧义 / 冲突 / 含糊修改\n进入临时确认队列", 0, 1, "planned"
            ),
            node(
                "ConfirmationItem", "临时事项结构\n字段 / 值 / 候选 / 约束 / 版本", 0, 2, "planned"
            ),
            node(
                "SemanticDecision",
                "模型理解接受 / 拒绝 / 修改\n支持部分或多项解决",
                0,
                3,
                "planned",
            ),
            node("UpdateDraft", "程序验证并更新草稿\n旧确认和令牌失效", 1, 3, "planned"),
            node("FinalPreview", "资料完整且无未解决事项\n生成新的完整预约预览", 1, 4, "planned"),
            node("SubmissionConfirmation", "scope = submission\n绑定草稿快照版本", 1, 5, "planned"),
            node("Submit", "确认版本有效且本轮无修改\n才允许业务提交", 1, 6, "planned"),
            node(
                "Modification", "最终确认中临时修改\n明确值直接更新 / 歧义再澄清", 2, 5, "planned"
            ),
        ],
        [
            edge("Collecting", "NeedsClarification", "需澄清", "planned"),
            edge("NeedsClarification", "ConfirmationItem", "", "planned"),
            edge("ConfirmationItem", "SemanticDecision", "", "planned"),
            edge("SemanticDecision", "UpdateDraft", "", "planned"),
            edge("UpdateDraft", "Collecting", "继续收集或处理队列", "planned"),
            edge("Collecting", "FinalPreview", "资料齐全有效", "planned"),
            edge("FinalPreview", "SubmissionConfirmation", "", "planned"),
            edge("SubmissionConfirmation", "Submit", "最终同意提交", "planned"),
            edge("SubmissionConfirmation", "Modification", "识别到修改", "planned"),
            edge("Modification", "UpdateDraft", "不消费旧确认", "planned"),
        ],
        [("PROJECT.md", "**3l：临时确认队列与最终确认")],
        "这是设计视图；当前没有通用 ConfirmationItem 队列。普通字段不逐个确认，修改优先于同句中的确认，模型语义准确率仍需真实多轮评测。",
        True,
    ),
]


# 主图以组件层级呈现派发关系，避免把所有回边挤到同一个行间通道。
main_view = VIEWS[1]
positions = {
    "ConversationGraph_invoke": (1.5, 0),
    "static_faq_cache": (1.5, 1),
    "supervisor": (1.5, 2),
    "AgentRegistry": (0, 2),
    "main_agent_response": (3, 2),
    "TaskCoordinator": (1.5, 3),
    "appointment_subgraph": (0, 4),
    "points_subgraph": (1, 4),
    "records_subgraph": (2, 4),
    "faq_subgraph": (3, 4),
    "ocr_subgraph": (3, 3),
    "main_agent_return": (1.5, 5),
    "END": (1.5, 6),
}
for item in main_view["nodes"]:
    item["col"], item["row"] = positions[item["id"]]
    if item["id"] == "TaskCoordinator":
        item["label"] = "领域派发关系\n单任务直派 / 多任务队列\nTaskCoordinator"
main_view["edges"] = [
    edge("ConversationGraph_invoke", "static_faq_cache"),
    edge("static_faq_cache", "supervisor", "未命中"),
    edge("static_faq_cache", "END", "缓存命中"),
    edge("AgentRegistry", "supervisor", "能力目录", "state"),
    edge("supervisor", "main_agent_response", "general"),
    edge("supervisor", "TaskCoordinator", "领域请求"),
    edge("TaskCoordinator", "ocr_subgraph", "暂缓注册项", "planned"),
    *[
        edge("TaskCoordinator", name, label)
        for name, label in [
            ("appointment_subgraph", "预约"),
            ("points_subgraph", "积分"),
            ("records_subgraph", "记录"),
            ("faq_subgraph", "FAQ"),
        ]
    ],
    *[
        edge(name, "main_agent_return", "交接")
        for name in ["appointment_subgraph", "points_subgraph", "records_subgraph", "faq_subgraph"]
    ],
    edge("main_agent_response", "END", "回复"),
    edge("main_agent_return", "END"),
]
main_view["guide"] += (
    " 图中领域派发框聚合两种路径：单任务由supervisor直派，多任务才经TaskCoordinator；不是所有请求都执行队列节点。"
)


def source_line(path, anchor):
    """定位来源行；找不到依据时停止生成，避免产出无来源关系。"""
    lines = (ROOT / path).read_text().splitlines()
    for number, line in enumerate(lines, 1):
        if anchor in line:
            return number
    raise ValueError(f"来源不存在: {path} / {anchor}")


def tool_catalog():
    """从实际 ToolSpec AST 生成工具目录，不维护第二份手写工具清单。"""
    path = "blood_agents/business/tools.py"
    tree = ast.parse((ROOT / path).read_text())
    names = []
    for item in ast.walk(tree):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "ToolSpec"
            and item.args
            and isinstance(item.args[0], ast.Constant)
        ):
            mutating = (
                len(item.args) > 3
                and isinstance(item.args[3], ast.Constant)
                and item.args[3].value is True
            )
            names.append({"name": item.args[0].value, "line": item.lineno, "mutating": mutating})
    return names


def catalog_view(tools):
    """为当前图谱来源中的工具生成组件目录。"""
    return view(
        "10-tool-catalog",
        "工具注册目录",
        f"本地集成｜{len(tools)} 项 ToolSpec，其中 OCR 暂缓；date_offset 是只读循环内置运算",
        [
            node(
                t["name"],
                t["name"]
                + "\n"
                + (
                    "OCR原型 / 暂缓"
                    if t["name"] == "ocr_extract"
                    else "需确认的写工具"
                    if t["mutating"]
                    else "查询 / 检索工具"
                ),
                i % 3,
                i // 3,
                "planned"
                if t["name"] == "ocr_extract"
                else "decision"
                if t["mutating"]
                else "tool",
            )
            for i, t in enumerate(tools)
        ],
        [],
        [("blood_agents/business/tools.py", "specs = [")],
        "这是组件目录，不表示这些工具并行执行。模型只读循环仅能看到显式 consultation 白名单，个人工具由服务端注入 user_id；写工具只能由领域确认流程调用。",
    )


def layout(v):
    """按声明式网格自动布局，给回边留出左右通道。"""
    width, height = (
        int(115 + (max(n["col"] for n in v["nodes"]) + 1) * 455),
        220 + (max(n["row"] for n in v["nodes"]) + 1) * 180,
    )
    for n in v["nodes"]:
        n.update(x=100 + n["col"] * 455, y=150 + n["row"] * 180, w=340, h=108)
    return width, height


def polyline(a, b, index, all_nodes, canvas_width):
    """生成共享正交路径；长边走网格外侧，减少穿过无关节点。"""
    ax, ay, bx, by = (
        a["x"] + a["w"] / 2,
        a["y"] + a["h"] / 2,
        b["x"] + b["w"] / 2,
        b["y"] + b["h"] / 2,
    )
    if (
        a["col"] == b["col"]
        and b["row"] > a["row"]
        and not any(n["col"] == a["col"] and a["row"] < n["row"] < b["row"] for n in all_nodes)
    ):
        return [(ax, a["y"] + a["h"]), (bx, b["y"])], "s", "n"
    if a["row"] == b["row"] and not any(
        n["row"] == a["row"] and min(a["col"], b["col"]) < n["col"] < max(a["col"], b["col"])
        for n in all_nodes
    ):
        return (
            ([(a["x"] + a["w"], ay), (b["x"], by)], "e", "w")
            if ax < bx
            else ([(a["x"], ay), (b["x"] + b["w"], by)], "w", "e")
        )
    if b["col"] == a["col"] + 1 and b["row"] == a["row"] - 1:
        return [(ax, a["y"]), (ax, by), (b["x"], by)], "n", "w"
    offset = (index % 4 - 1.5) * 9
    if b["row"] == a["row"] + 1:
        y = (a["y"] + a["h"] + b["y"]) / 2 + offset
        return [(ax, a["y"] + a["h"]), (ax, y), (bx, y), (bx, b["y"])], "s", "n"
    # 回边及跨多行连线，使用页面边缘，不穿过中间节点。
    lane = (
        45 + index % 4 * 12
        if b["row"] < a["row"] or b["col"] <= a["col"]
        else canvas_width - 50 - index % 4 * 12
    )
    y1 = a["y"] + a["h"] + 35 + offset
    y2 = b["y"] - 35 - offset
    return (
        [(ax, a["y"] + a["h"]), (ax, y1), (lane, y1), (lane, y2), (bx, y2), (bx, b["y"])],
        "s",
        "n",
    )


def edge_color(e):
    """按控制、存储和错误关系给连线着色；不把存储访问误画成推测关系。"""
    return {"state": "#8867B4", "error": "#CA6B5A", "planned": "#87909D"}.get(e["kind"], "#738397")


def lines_for(text, font, width):
    """按实际字体宽度折行，避免中文和长符号被节点边界裁切。"""
    output = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            if line and font.getlength(line + char) > width:
                output.append(line)
                line = ""
            line += char
        output.append(line)
    return output


def render(v):
    """生成 draw.io 原生元素和共享几何的离线 SVG/PNG 预览。"""
    from PIL import Image, ImageDraw, ImageFont

    width, height = layout(v)
    font_path = next(
        (
            str(p)
            for p in [
                Path(os.getenv("BLOOD_DIAGRAM_FONT", "/System/Library/Fonts/STHeiti Medium.ttc")),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ]
            if p.is_file()
        ),
        "",
    )
    if not font_path:
        raise RuntimeError("请用 BLOOD_DIAGRAM_FONT 指定中文字体路径")
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 14)
    title = ImageFont.truetype(font_path, 30)
    im = Image.new("RGB", (width, height), "#F7F9FC")
    d = ImageDraw.Draw(im)
    d.text((100, 38), v["title"], font=title, fill="#14263D")
    d.text((100, 88), v["subtitle"], font=small, fill="#586B82")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F9FC"/>',
        f'<text x="100" y="66" font-family="sans-serif" font-size="30" fill="#14263D">{html.escape(v["title"])}</text>',
        f'<text x="100" y="106" font-family="sans-serif" font-size="14" fill="#586B82">{html.escape(v["subtitle"])}</text>',
    ]
    page = ET.Element("diagram", id=v["id"], name=v["title"])
    model = ET.SubElement(
        page,
        "mxGraphModel",
        dx="1480",
        dy=str(height),
        grid="1",
        gridSize="10",
        page="1",
        pageScale="1",
        pageWidth=str(width),
        pageHeight=str(height),
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    nodes = {n["id"]: n for n in v["nodes"]}
    for i, e in enumerate(v["edges"]):
        assert e["source"] in nodes and e["target"] in nodes
        pts, start, end = polyline(nodes[e["source"]], nodes[e["target"]], i, nodes.values(), width)
        color = edge_color(e)
        d.line(pts, fill=color, width=2)
        p, q = pts[-2:]
        angle = math.atan2(q[1] - p[1], q[0] - p[0])
        arr = [
            q,
            (q[0] - 11 * math.cos(angle - 0.45), q[1] - 11 * math.sin(angle - 0.45)),
            (q[0] - 11 * math.cos(angle + 0.45), q[1] - 11 * math.sin(angle + 0.45)),
        ]
        d.polygon(arr, fill=color)
        dashed = ' stroke-dasharray="7 5"' if e["kind"] == "planned" else ""
        svg.append(
            f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="2"{dashed}/><polygon points="{" ".join(f"{x},{y}" for x, y in arr)}" fill="{color}"/>'
        )
        if e["label"]:
            seg = max(
                itertools.pairwise(pts),
                key=lambda pq: abs(pq[1][0] - pq[0][0]) + abs(pq[1][1] - pq[0][1]),
            )
            lx, ly = (seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2
            tw = small.getlength(e["label"])
            d.rectangle((lx - tw / 2 - 5, ly - 12, lx + tw / 2 + 5, ly + 10), fill="#F7F9FC")
            d.text((lx - tw / 2, ly - 9), e["label"], font=small, fill="#4F6279")
            svg.append(
                f'<rect x="{lx - tw / 2 - 5}" y="{ly - 12}" width="{tw + 10}" height="23" fill="#F7F9FC"/><text x="{lx}" y="{ly + 5}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#4F6279">{html.escape(e["label"])}</text>'
            )
        ports = {"s": (0.5, 1), "n": (0.5, 0), "e": (1, 0.5), "w": (0, 0.5)}
        sx, sy = ports[start]
        tx, ty = ports[end]
        cell = ET.SubElement(
            root,
            "mxCell",
            id=f"{v['id']}_edge_{i}",
            value=e["label"],
            edge="1",
            parent="1",
            source=e["source"],
            target=e["target"],
            style=f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;strokeColor={color};fontSize=13;labelBackgroundColor=#F7F9FC;exitX={sx};exitY={sy};entryX={tx};entryY={ty};"
            + ("dashed=1;" if e["kind"] == "planned" else ""),
        )
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        arrxml = ET.SubElement(geom, "Array", **{"as": "points"})
        for x, y in pts[1:-1]:
            ET.SubElement(arrxml, "mxPoint", x=str(x), y=str(y))
    for n in v["nodes"]:
        fill, stroke = PALETTE[n["kind"]]
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        wrapped = lines_for(n["label"], font, w - 28)
        if len(wrapped) > 4:
            raise ValueError(f"节点文字过长:{v['id']}/{n['id']}")
        d.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill, outline=stroke, width=2)
        for j, line in enumerate(wrapped):
            lx = x + (w - font.getlength(line)) / 2
            ly = y + (h - len(wrapped) * 24) / 2 + j * 24
            d.text((lx, ly), line, font=font, fill="#24364C")
        dash = ' stroke-dasharray="7 5"' if n["kind"] == "planned" else ""
        svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        for j, line in enumerate(wrapped):
            ly = y + (h - len(wrapped) * 24) / 2 + j * 24 + 19
            svg.append(
                f'<text x="{x + w / 2}" y="{ly}" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#24364C">{html.escape(line)}</text>'
            )
        cell = ET.SubElement(
            root,
            "mxCell",
            id=n["id"],
            value=n["label"],
            vertex="1",
            parent="1",
            style=f"rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontColor=#24364C;fontSize=18;spacing=12;"
            + ("dashed=1;" if n["kind"] == "planned" else ""),
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(w),
            height=str(h),
            **{"as": "geometry"},
        )
    footer = (
        SNAPSHOT_DATE
        + " · "
        + ("待实现设计" if v["planned"] else "本地集成快照")
        + " · 离线共享几何预览；可编辑源见 smart-blood-donation.drawio"
    )
    d.text((100, height - 35), footer, font=small, fill="#6B7C90")
    svg.append(
        f'<text x="100" y="{height - 20}" font-family="sans-serif" font-size="14" fill="#6B7C90">{html.escape(footer)}</text></svg>'
    )
    (OUT / (v["id"] + ".svg")).write_text("\n".join(svg))
    im.save(OUT / (v["id"] + ".png"))
    return page


def mermaid(v):
    """生成与当前页同源的 Mermaid；每页明确区分已有和规划内容。"""
    lines = [
        "%%{init: {'theme': 'base', 'flowchart': {'curve': 'linear'}, 'themeVariables': {'fontFamily': 'sans-serif', 'fontSize': '15px'}}}%%",
        "flowchart TB",
    ]
    for kind, (fill, stroke) in PALETTE.items():
        lines.append(
            f"    classDef {kind} fill:{fill},stroke:{stroke},color:#24364C"
            + (",stroke-dasharray:5 3" if kind == "planned" else "")
            + ";"
        )
    for n in v["nodes"]:
        lines.append(f'    {n["id"]}["{n["label"].replace(chr(10), "<br/>")}"]:::{n["kind"]}')
    for e in v["edges"]:
        arrow = "-.->" if e["kind"] == "planned" else "-->"
        label = f'|"{e["label"]}"|' if e["label"] else ""
        lines.append(f"    {e['source']} {arrow}{label} {e['target']}")
    if v["edges"]:
        for i, e in enumerate(v["edges"]):
            lines.append(f"    linkStyle {i} stroke:{edge_color(e)},stroke-width:1.5px;")
    return "\n".join(lines) + "\n"


def main():
    """默认从已核对的快照重建，完整集成工作区可重新核对代码来源。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-source", action="store_true", help="重新核对集成工作区的源码锚点")
    args = parser.parse_args()
    snapshot = {} if args.from_source else json.loads((OUT / "sources.json").read_text())
    tools = tool_catalog() if args.from_source else snapshot["tools"]
    saved_lines = {
        (ref["path"], ref["anchor"]): ref["line"]
        for item in snapshot.get("views", []) + snapshot.get("sequences", [])
        for ref in item["sources"]
    }

    def references(items):
        """完整工作区核对锚点，文档检出使用已记录的来源行。"""
        return [
            {
                "path": path,
                "anchor": anchor,
                "line": source_line(path, anchor)
                if args.from_source
                else saved_lines[path, anchor],
            }
            for path, anchor in items
        ]

    views = [*VIEWS[:9], catalog_view(tools), *VIEWS[9:]]
    mx = ET.Element(
        "mxfile",
        host="app.diagrams.net",
        version="26.0.0",
        modified=f"{SNAPSHOT_DATE}T00:00:00.000Z",
    )
    readme = [
        "# 架构图与流程图",
        "",
        f"更新于 {SNAPSHOT_DATE}。{SCOPE}第 12 页为待实现设计。",
        "",
        "[可编辑 draw.io](smart-blood-donation.drawio) · [离线图册](index.html) · [系统说明](../ARCHITECTURE.md) · [构建验证](VALIDATION.md)",
        "",
        "PNG/SVG 是同一份节点、文字和正交几何生成的离线预览，**不是 draw.io 原生导出**。Mermaid 为可维护文本源；没有依赖在线 CDN。",
        "",
        "蓝色为编排，绿色为领域/工具，紫色为模型/存储，橙色为条件/外部服务，灰色虚线为暂缓或规划。实线关系有代码来源；图中聚合节点代表同一职责的组件集合。",
        "",
        "| 页 | 内容 | 文件 |",
        "| --- | --- | --- |",
    ]
    manifest = []
    for v in views:
        if len(v["nodes"]) > 30:
            raise ValueError("请拆分大图")
        if len({n["id"] for n in v["nodes"]}) != len(v["nodes"]):
            raise ValueError("重复节点")
        refs = references(v["refs"])
        mx.append(render(v))
        (OUT / (v["id"] + ".mmd")).write_text(mermaid(v))
        readme.append(
            f"| {v['id'][:2]} | {v['title']} | [PNG]({v['id']}.png) / [SVG]({v['id']}.svg) / [Mermaid]({v['id']}.mmd) |"
        )
        manifest.append(
            {k: v[k] for k in ["id", "title", "subtitle", "planned", "guide"]}
            | {"sources": refs, "node_count": len(v["nodes"]), "edge_count": len(v["edges"])}
        )
    readme += ["", "## 阅读与来源", ""]
    for v, m in zip(views, manifest):
        readme += [
            f"### {v['title']}",
            "",
            v["guide"],
            "",
            "本地快照源码位置 "
            + "；".join(f"`{r['path']}:{r['line']}`" for r in m["sources"])
            + "。",
            "",
        ]
    sequences_doc = [
        "# 运行时序图",
        "",
        f"{SCOPE}流程图和可编辑源见 [图册索引](README.md)。",
        "",
    ]
    sequence_manifest = []
    for seq in SEQUENCES:
        (OUT / (seq["id"] + ".mmd")).write_text(seq["text"])
        refs = references(seq["sources"])
        sequence_manifest.append({"id": seq["id"], "sources": refs})
        readme += [
            "### " + seq["title"],
            "",
            seq["guide"],
            "",
            "[Mermaid时序源](" + seq["id"] + ".mmd) · [内嵌预览](SEQUENCES.md)",
            "",
        ]
        sequences_doc += [
            "## " + seq["title"],
            "",
            seq["guide"],
            "",
            "```mermaid",
            seq["text"].rstrip(),
            "```",
            "",
            "本地快照源码位置 " + "；".join(f"`{r['path']}:{r['line']}`" for r in refs),
            "",
        ]
    (OUT / "SEQUENCES.md").write_text("\n".join(sequences_doc).rstrip() + "\n")
    ET.indent(mx, space="  ")
    ET.ElementTree(mx).write(
        OUT / "smart-blood-donation.drawio", encoding="utf-8", xml_declaration=True
    )
    (OUT / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    (OUT / "sources.json").write_text(
        json.dumps(
            {
                "generated": SNAPSHOT_DATE,
                "framework": "LangGraph",
                "views": manifest,
                "tools": tools,
                "sequences": sequence_manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    nav = "".join(
        f'<a href="#{v["id"]}">{i + 1:02d} {html.escape(v["title"])}</a>'
        for i, v in enumerate(views)
    )
    body = "".join(
        f'<section id="{v["id"]}"><h2>{html.escape(v["title"])}</h2><p>{html.escape(v["guide"])}</p><a href="{v["id"]}.svg"><img loading="lazy" src="{v["id"]}.svg" alt="{html.escape(v["title"])}"/></a><p><a href="{v["id"]}.png">PNG</a> · <a href="{v["id"]}.mmd">Mermaid源</a></p></section>'
        for v in views
    )
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>智慧献血 · 架构图册</title><style>body{margin:0;background:#f7f9fc;color:#24364c;font:16px/1.7 system-ui,sans-serif}nav{position:fixed;inset:0 auto 0 0;width:240px;padding:24px;background:white;border-right:1px solid #e1e7ef;overflow:auto}nav a{display:block;padding:7px 0}a{color:#2a6bb4;text-decoration:none}main{margin-left:290px;padding:30px}section{background:white;border:1px solid #e1e7ef;border-radius:14px;padding:24px;margin-bottom:28px;scroll-margin-top:20px}img{width:100%;height:auto}h1{font-size:28px}h2{font-size:22px}@media(max-width:800px){nav{position:static;width:auto}main{margin:0;padding:14px}}</style><nav><h1>智慧献血</h1>'
        + f"<p>架构与流程 · {SNAPSHOT_DATE}</p>"
        + nav
        + '<p><a href="smart-blood-donation.drawio">下载可编辑图册</a></p></nav><main><p>'
        + html.escape(SCOPE)
        + "灰色规划页未实现。PNG/SVG 为离线共享几何预览。</p>"
        + body
        + "</main></html>"
    )
    # 保持旧文档引用可用，但替换成当前对应视图，避免残留过时图片。
    aliases = {
        "architecture-preview.png": "01-system-context.png",
        "appointment-flow-preview.png": "03-appointment.png",
        "target-architecture-preview.png": "12-planned-confirmation.png",
    }
    for old, new in aliases.items():
        (OUT / old).write_bytes((OUT / new).read_bytes())
    print(
        json.dumps(
            {
                "pages": len(views),
                "tools": len(tools),
                "source_mode": "checkout" if args.from_source else "snapshot",
                "sequences": len(SEQUENCES),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
