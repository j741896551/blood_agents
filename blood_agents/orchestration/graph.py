"""LangGraph 接线：主导航派发领域子图，预约与积分子图可跨轮等待。"""

from __future__ import annotations

import os
import re
import uuid
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from blood_agents.business.legacy_api import LegacyBloodAPI
from blood_agents.business.tools import build_business_tools, tool_trace
from blood_agents.core.models import AgentState
from blood_agents.domains.ocr_service import OCRService
from blood_agents.knowledge.donation_interval import DonationIntervalPlanner
from blood_agents.knowledge.evidence import EvidenceAssessor
from blood_agents.knowledge.rag import HybridRetriever, default_retriever
from blood_agents.knowledge.static_faq_cache import StaticFaqCache
from blood_agents.llm.reply_composer import ReplyComposer
from blood_agents.llm.tool_planner import AppointmentToolPlanner
from blood_agents.log.trace_logging import current_trace, trace_event
from blood_agents.storage.checkpoint import SQLAlchemySqliteSaver
from blood_agents.storage.repository import AgentRepository

from .agent_registry import AgentDependencies, AgentSpec
from .conversation import ConversationGraph
from .long_term_memory import LongTermMemoryService
from .router import IntentRouter
from .short_term_memory import project_short_term_memory
from .user_profile import is_profile_query, is_profile_signal


def _legacy_api_from_env(repo: AgentRepository) -> LegacyBloodAPI:
    """从外部 HTTP/MCP 适配器或可选本地业务插件取得接口实现。"""
    transport = os.getenv("BLOOD_AGENT_BUSINESS_TRANSPORT", "http").strip().lower()
    if transport == "local":
        # 本地业务替身不随 Agent 包提交；缺失时明确报错，不伪装为空回执。
        try:
            factory = import_module("local_business.api").create_local_api
        except ImportError as exc:
            raise RuntimeError("本地业务插件未安装；请改用外部 HTTP/MCP 接口") from exc
        return factory(repo)
    if transport in {"http", "mcp"}:
        # Agent 侧仅保留接口客户端；服务端实现属于外部业务系统。
        from blood_agents.business.legacy_http import LegacyHTTPClient

        base_url = os.getenv("BLOOD_AGENT_LEGACY_API_URL", "http://127.0.0.1:8001")
        timeout = float(os.getenv("BLOOD_AGENT_LEGACY_API_TIMEOUT", "10"))
        return LegacyHTTPClient(base_url, timeout=timeout)
    raise ValueError("BLOOD_AGENT_BUSINESS_TRANSPORT 仅支持 local、http 或 mcp")


def build_graph(
    repo: Optional[AgentRepository] = None,
    router: Optional[IntentRouter] = None,
    retriever: Optional[HybridRetriever] = None,
    ocr: Optional[OCRService] = None,
    legacy_api: Optional[LegacyBloodAPI] = None,
    checkpointer=None,
    mcp_queries=None,
    static_cache: Optional[StaticFaqCache] = None,
):
    """组装主图和领域子图；接口、模型、检索器均可注入以隔离测试。

    调用顺序：静态问答缓存 → 未命中时 supervisor → 领域子图
    （理解/可选工具规划/执行/回复）→ 等待用户或 main_agent_return。
    构图只接线和注入依赖；业务接口在请求进入执行节点后才调用。

    checkpointer 安装在父图上，供跨轮恢复预约与积分状态；测试可注入
    会话仓储、模型路由和检索器；业务事实由注入的外部接口提供。
    """
    repo = repo or AgentRepository()
    router = router or IntentRouter.from_env()
    registry = router.registry
    # 编译后的图不会随注册表变更而自动接线，因此启动时冻结同一份 Agent 清单。
    registry.freeze()
    retriever = retriever or default_retriever()
    ocr = ocr or OCRService()
    legacy_api = legacy_api or _legacy_api_from_env(repo)
    tools = build_business_tools(repo, retriever, ocr, legacy_api, mcp_queries=mcp_queries)
    evidence_assessor = (EvidenceAssessor(router.evidence_llm)
                         if router.evidence_llm else None)
    interval_planner = DonationIntervalPlanner(router.tool_llm)
    dependencies = AgentDependencies(repo, legacy_api, retriever, ocr, tools,
                                     evidence_assessor, interval_planner)
    reply_composer = ReplyComposer(router.response_llm, router.risk_llm)
    appointment_planner = AppointmentToolPlanner(router.tool_llm)
    long_term_memory = LongTermMemoryService(repo, router.llm)
    static_cache = static_cache if static_cache is not None else StaticFaqCache.from_file()

    def cache_lookup(state: AgentState) -> dict:
        """主导航前的窄门：唯一静态命中直接回答，不执行模型或业务工具。"""
        question = state["messages"][-1].content
        entry, reason = static_cache.lookup(question, active_agent=state.get("active_agent"))
        trace_event("cache.lookup", summary={
            "status": "hit" if entry else "miss", "reason": reason,
            "cache_key": entry.key if entry else None,
            "cache_version": entry.version if entry else None,
        })
        if entry is None:
            return {"cache_info": {"status": "miss", "reason": reason}}
        return {
            "messages": [AIMessage(content=entry.answer)],
            "intent": "general",
            "result": {"status": "static_faq_cache", "cache_key": entry.key},
            "response_info": {"mode": "static_cache"},
            "model_info": {"mode": "skipped_static_cache"},
            "cache_info": {"status": "hit", "key": entry.key, "version": entry.version},
            "tool_trace": [],
            "node_trace": [{"node": "static_faq_cache", "status": "hit", "duration_ms": 0}],
        }

    def traced(name, node):
        """为图节点包裹耗时与工具轨迹记录。"""
        def run(state):
            """运行被追踪的图节点并附加耗时和工具轨迹。"""
            trace = []
            started = monotonic()
            token = tool_trace.set(trace)
            trace_event("graph.node.start", summary={"node": name})
            try:
                result = node(state)
                failed = bool(result.get("error")) or result.get("result", {}).get("success") is False
                entry = {"node": name, "status": "failed" if failed else "success",
                         "duration_ms": round((monotonic() - started) * 1000, 2)}
                trace_event("graph.node.end", summary={
                    **entry, "result_status": result.get("result", {}).get("status"),
                    "tool_count": len(trace),
                })
                # 后续回复节点不调用工具，也不能因此清掉业务节点已有的工具轨迹。
                return {**result, "tool_trace": [*state.get("tool_trace", []), *trace],
                        "node_trace": [*state.get("node_trace", []), entry]}
            except Exception as exc:
                trace_event("graph.node.error", summary={
                    "node": name, "error_type": type(exc).__name__,
                    "duration_ms": round((monotonic() - started) * 1000, 2),
                })
                raise
            finally:
                tool_trace.reset(token)
        return run

    def supervisor(state: AgentState) -> dict:
        """识别本轮用户意图并选择领域子图。"""
        started = monotonic()
        # 主导航只负责领域分派。预约参数和确认动作在预约子图理解，
        # 防止主图承担所有领域的内部业务语义。
        content = state["messages"][-1].content
        recent = [
            {"role": message.type, "content": message.content}
            for message in state["messages"][-5:-1]
        ]
        # 每轮从 checkpoint 的权威流程字段重建快照，忽略旧摘要或模型推断。
        memory = project_short_term_memory(state)
        memory_event = {"status": "not_applicable"}
        if state.get("session_id"):
            trace = current_trace.get()
            memory_event = long_term_memory.capture(
                user_id=state["user_id"], session_id=state["session_id"],
                trace_id=trace.trace_id if trace else uuid.uuid4().hex, text=content,
            )
        try:
            preferences = [
                {"key": item["preference_key"], "value": item["value"],
                 "source": "explicit_user_statement", "source_quote": item["source_quote"]}
                for item in repo.list_active_preferences(state["user_id"])
            ]
        except Exception as exc:  # noqa: BLE001 - 记忆读取失败不能阻断业务图
            preferences = []
            trace_event("memory.load_failed", summary={"error_type": type(exc).__name__})
        profile_query = is_profile_query(content)
        profile_status = "not_requested"
        try:
            # 画像只含兴趣字段；失败时按空画像处理，不影响导航和业务工具。
            user_profile = repo.get_user_profile(state["user_id"])
            if profile_query:
                # 入口后台更新若曾失败，不能把尚未汇总的对话误说成“没有爱好”。
                profile_cursor = repo.get_user_profile_cursor(state["user_id"])
                pending_turns = repo.list_user_conversation_turns_after(
                    state["user_id"], profile_cursor, limit=1,
                )
                profile_status = "pending_update" if pending_turns else "ready"
        except Exception as exc:  # noqa: BLE001 - 画像读取失败不阻断主图
            user_profile = {"hobbies": []}
            profile_status = "unavailable" if profile_query else "not_requested"
            trace_event("user_profile.load_failed", summary={"error_type": type(exc).__name__})
        trace_event("memory.short_term", summary={
            "active_agent": memory["active_agent"],
            "current_task": (memory["current_task"] or {}).get("agent"),
            "suspended_task_count": len(memory["suspended_tasks"]),
        }, full={"snapshot": memory})
        interpretation, info = router.navigate(
            content, state.get("active_agent"), recent, short_term_memory=memory,
        )
        # 模型把不完整的预约补充语句判为 general 时，仍续接活动预约。
        # 明确问候以及其他领域问题可由主导航或相应子图处理，不清除预约状态。
        if (state.get("active_agent") == "appointment" and interpretation.intent == "general"
                and not router.is_general_turn(content) and not is_profile_signal(content)):
            trace_event("navigation.guardrail", summary={
                "from_intent": "general", "to_intent": "appointment",
                "reason": "continue_active_appointment",
            })
            interpretation = interpretation.model_copy(update={"intent": "appointment"})
        if (state.get("active_agent") == "points" and interpretation.intent == "general"
                and not router.is_general_turn(content) and not is_profile_signal(content)):
            interpretation = interpretation.model_copy(update={"intent": "points"})
        # 单独说“记住/忘记偏好”不是新预约。活动业务流程仍在 checkpoint 中等待续接。
        preference_only = (
            long_term_memory.is_preference_query(content) or
            is_profile_signal(content) or
            (memory_event["status"] != "not_applicable" and not re.search(
                r"预约|兑换|查询|查看|查一下|取消订单|签到|献血记录", content,
            ))
        )
        if preference_only:
            interpretation = interpretation.model_copy(update={"intent": "general"})
        intent = interpretation.intent
        trace_event("graph.dispatch", summary={
            "intent": intent, "active_agent": state.get("active_agent"),
            "model_mode": info.get("mode"),
        })
        c =  {
            "intent": intent,
            "understanding": interpretation.model_dump(exclude_none=True),
            "model_info": {**info, "navigation": info},
            "response_info": {},
            "appointment_tool_data": {},
            "tool_plan_info": {},
            "agent_status": None,
            "agent_handoff": {},
            "short_term_memory": memory,
            "long_term_preferences": preferences,
            "user_profile": user_profile,
            "profile_query": profile_query,
            "profile_status": profile_status,
            "memory_event": memory_event,
            "tool_trace": [],
            "error": "",
            "node_trace": [{"node": "supervisor", "status": "success",
                            "duration_ms": round((monotonic() - started) * 1000, 2)}],
        }
        return c

    def main_agent_response(state: AgentState) -> dict:
        """问候和非领域请求由主导航直接回答，不触发知识库或业务工具。"""
        trace_event("graph.node.end", summary={"node": "main_agent_response", "status": "success"})
        return {
            "result": {
                "status": "general",
                # OCR 仍保留原型和注册项，但用户已暂缓该阶段，不能在问候中将它
                # 宣传为当前可用的完整能力。
                "capabilities": [spec.description for spec in registry.specs()
                                 if spec.intent != "ocr"],
                # 用户问系统自身的部署/数据来源时，由主导航使用明确的工程事实；
                # 不让没有检索证据的模型猜测是否接入公司旧接口。
                "service_facts": {
                    "business_transport": os.getenv("BLOOD_AGENT_BUSINESS_TRANSPORT", "http"),
                    "business_connection_verified": False,
                    "ocr_stage": "deferred_prototype_only",
                },
                "active_appointment": state.get("active_agent") == "appointment",
            },
            "node_trace": [*state.get("node_trace", []), {
                "node": "main_agent_response", "status": "success", "duration_ms": 0,
            }],
        }

    def appointment_subgraph_start(state: AgentState) -> dict:
        """为预约子图写入可观察的派发节点；业务槽位仍由已有预约编排器管理。"""
        trace_event("graph.node.start", summary={"node": "appointment_subgraph"})
        return {
            # appointment_state 是预约子 Agent 的权威状态；同步到 slots 是为
            # 保持现有 UI、API 响应和教学示例的兼容，避免两份状态分别演进。
            "slots": dict(state.get("appointment_state") or state.get("slots", {})),
            "node_trace": [*state.get("node_trace", []), {
                "node": "appointment_subgraph",
                "status": "started",
                "duration_ms": 0,
            }],
        }

    def appointment_understand(state: AgentState) -> dict:
        """预约子图自己的模型节点；只补充槽位/动作，不负责导航和接口结果。"""
        content = state["messages"][-1].content
        recent = [
            {"role": message.type, "content": message.content}
            for message in state["messages"][-5:-1]
        ]
        interpretation, info = router.interpret_appointment(
            content,
            slots=state.get("slots"),
            availability=state.get("result", {}).get("availability", []),
            recent_messages=recent,
        )
        trace_event("appointment.understood", summary={
            "mode": info.get("mode"), "action": interpretation.action,
            "slot_keys": list(interpretation.slots.model_dump(exclude_none=True)),
        }, full={"slots": interpretation.slots.model_dump(exclude_none=True)})
        navigation = state.get("model_info", {}).get("navigation", {})
        mode = "rules_fallback" if "rules_fallback" in (navigation.get("mode"), info.get("mode")) else info["mode"]
        attempts = [
            {"stage": stage, **attempt}
            for stage, stage_info in (("navigation", navigation), ("appointment", info))
            for attempt in stage_info.get("attempts", [])
        ]
        return {
            "understanding": {"intent": "appointment", **interpretation.model_dump(exclude_none=True),
                              "rejected_sites": info.get("rejected_sites", [])},
            "model_info": {
                "mode": mode,
                "provider": info.get("provider") or navigation.get("provider"),
                "error_type": info.get("error_type") or navigation.get("error_type"),
                "attempts": attempts,
                "navigation": navigation,
                "appointment": info,
            },
        }

    def appointment_handoff(state: AgentState) -> dict:
        """区分等待用户的中间态与可交还主图的预约终态。"""
        result = dict(state.get("result") or {})
        slots = dict(state.get("slots") or {})
        status = result.get("status")
        # 待确认状态不是业务接口的事实，而是“已经向用户展示预览”的授权边界。
        # 模型回复失败时页面只显示服务不可用；此时保留确认状态会让下一轮短句
        # 直接触发预约/取消/签到写接口。旧 checkpoint 也必须有本标记才可确认。
        if status in {"awaiting_confirmation", "awaiting_action_confirmation"}:
            delivered = (state.get("response_info") or {}).get("mode") == "llm"
            if delivered:
                slots["preview_delivered"] = True
            else:
                slots.pop("preview_delivered", None)
                if status == "awaiting_confirmation":
                    slots.pop("slot_id", None)
                    slots.pop("donor_confirmed", None)
                    slots["status"] = "collecting"
                else:
                    for key in ("pending_operation", "pending_appointment_id", "status"):
                        slots.pop(key, None)
                    if slots.get("flow") == "appointment":
                        slots["status"] = "collecting"
                result = {"status": "preview_unavailable", "success": False,
                          "message": "待确认预览未送达，写操作未执行；请重新发起并核对预览。"}
                status = result["status"]
                trace_event("appointment.preview_discarded", summary={
                    "reason": "reply_unavailable", "preview_type": (
                        "booking" if state.get("result", {}).get("status") == "awaiting_confirmation"
                        else "existing_appointment_action"),
                })
        waiting_statuses = {"need_more_info", "slot_unavailable", "slot_changed", "awaiting_confirmation",
                            "awaiting_action_confirmation"}
        # 查询已提交预约不应终止另一个仍在收集/待确认的预约流程。
        query_statuses = {"appointments_found", "no_appointments", "query_failed",
                          "cancel_unavailable"}
        waiting = status in waiting_statuses or (
            status in query_statuses and slots.get("status") in
            {"collecting", "awaiting_confirmation"}
        ) or (status == "preview_unavailable" and slots.get("status") == "collecting")
        agent_status = "waiting" if waiting else "completed"
        trace_event("agent.handoff", summary={
            "agent": "appointment", "status": status or agent_status,
            "control": "user" if waiting else "main",
        })
        update = {
            "active_agent": "appointment" if waiting else None,
            "agent_status": agent_status,
            "slots": slots,
            "result": result,
            "appointment_state": slots,
            "agent_handoff": {
                "agent": "appointment",
                "status": status or agent_status,
                "control": "user" if waiting else "main",
            },
            "node_trace": [*state.get("node_trace", []), {
                "node": "appointment_handoff",
                "status": agent_status,
                "duration_ms": 0,
            }],
        }
        # 回复之后再刷新本轮快照，避免把已完成/已退出任务留给下一轮导航。
        update["short_term_memory"] = project_short_term_memory({**state, **update})
        return update

    def appointment_tool_plan(state: AgentState) -> dict:
        """模型只列出允许的只读工具；入口用户身份由这里注入，不采信模型参数。"""
        action = state.get("understanding", {}).get("action", "collect")
        message = next((item.content for item in reversed(state["messages"])
                        if getattr(item, "type", None) == "human"), "")
        names, info = appointment_planner.plan(message, action, state.get("slots", {}))
        data = {name: tools.invoke(name, user_id=state.get("user_id", "U1001")) for name in names}
        return {"appointment_tool_data": data,
                "tool_plan_info": {**info, "requested_tools": names}}

    def points_understand(state: AgentState) -> dict:
        """积分子图自行判断查询、兑换意向、确认或退出；不由主图执行写操作。"""
        content = state["messages"][-1].content
        recent = [{"role": message.type, "content": message.content}
                  for message in state["messages"][-5:-1]]
        decision, info = router.interpret_points(
            content, pending=state.get("points_state"), recent_messages=recent,
        )
        navigation = state.get("model_info", {}).get("navigation", {})
        mode = ("rules_fallback" if "rules_fallback" in
                (navigation.get("mode"), info.get("mode")) else info["mode"])
        trace_event("points.understood", summary={
            "mode": info.get("mode"), "action": decision.action,
            "reward_id": decision.reward_id,
        })
        return {
            "understanding": {"intent": "points", **decision.model_dump(exclude_none=True)},
            "model_info": {"mode": mode, "provider": info.get("provider") or navigation.get("provider"),
                           "error_type": info.get("error_type") or navigation.get("error_type"),
                           "navigation": navigation, "points": info},
        }

    def points_handoff(state: AgentState) -> dict:
        """待确认兑换留在积分子图；完成/退出后把控制权交还主导航。"""
        pending = dict(state.get("points_state") or {})
        # 预览未由可用模型交付给用户时不能保留可确认的写操作状态。
        if pending and state.get("result", {}).get("status") in {
            "awaiting_redemption_confirmation", "awaiting_points_confirmation", "address_details_required",
            "redemption_address_required", "address_selection_required",
        }:
            model_info = state.get("model_info") or {}
            modes = ((model_info.get("navigation") or {}).get("mode"),
                     (model_info.get("points") or {}).get("mode"),
                     (state.get("response_info") or {}).get("mode"))
            if modes != ("llm", "llm", "llm"):
                pending = {}
                trace_event("points.preview_discarded", summary={
                    "reason": "preview_or_understanding_unavailable", "model_modes": modes,
                })
        waiting = bool(pending)
        # 积分只读查询可临时打断未完成的预约；回交主图后原预约仍可继续。
        resume_appointment = (
            not waiting and state.get("active_agent") == "appointment"
            and (state.get("appointment_state") or {}).get("status") in
            {"collecting", "awaiting_confirmation"}
        )
        status = (state.get("result") or {}).get("status", "completed")
        trace_event("agent.handoff", summary={
            "agent": "points", "status": status,
            "control": "user" if waiting else "main",
        })
        update = {
            "points_state": pending,
            "active_agent": "points" if waiting else "appointment" if resume_appointment else None,
            "agent_status": "waiting" if waiting else "completed",
            "agent_handoff": {"agent": "points", "status": status,
                              "control": "user" if waiting else "main"},
            "node_trace": [*state.get("node_trace", []), {
                "node": "points_handoff", "status": "waiting" if waiting else "completed",
                "duration_ms": 0,
            }],
        }
        update["short_term_memory"] = project_short_term_memory({**state, **update})
        return update

    def main_agent_return(state: AgentState) -> dict:
        """主图接收领域 Agent 的执行结果，再将本轮结果交给入口展示。"""
        handoff = dict(state.get("agent_handoff", {}))
        if not handoff:
            handoff = {
                "agent": state.get("intent", "unknown"),
                "status": "completed",
            }
        handoff["control"] = "main"
        trace_event("agent.handoff", summary={
            "agent": handoff.get("agent"), "status": handoff.get("status"), "control": "main",
        })
        return {
            "agent_handoff": handoff,
            "node_trace": [*state.get("node_trace", []), {
                "node": "main_agent_return",
                "status": "success",
                "duration_ms": 0,
            }],
        }

    def reply_node(intent: str):
        """领域内的独立回复节点：只让模型表达结构化事实，不触碰工具与流程状态。"""
        def node(state: AgentState) -> dict:
            """调用领域回复模型，把结构化事实转为用户答复。"""
            answer, info = reply_composer.compose(intent, state)
            # 内置领域节点不再生成业务答复。兼容注册式外部 Agent 的旧草稿，
            # 即使模型不可用，也只会被统一错误信息覆盖，不会展示草稿。
            draft_message = state["messages"][-1]
            message_id = draft_message.id if isinstance(draft_message, AIMessage) else None
            return {
                "messages": [AIMessage(content=answer, id=message_id)],
                "response_info": info,
            }
        return node

    def build_appointment_subgraph(spec: AgentSpec, agent_node):
        """编译预约多轮子图：理解、只读规划、执行、回复和交接。"""
        # 预约有自己的跨轮状态机；注册信息只负责接线，
        # 确认后提交和等待用户等语义仍由原有预约流程控制。
        appointment_builder = StateGraph(AgentState)
        appointment_builder.add_node("appointment_subgraph_start", appointment_subgraph_start)
        appointment_builder.add_node(
            "appointment_understand", traced("appointment_understand", appointment_understand),
        )
        appointment_builder.add_node("appointment_tool_plan", traced("appointment_tool_plan", appointment_tool_plan))
        appointment_builder.add_node(spec.agent_name, traced(spec.agent_name, agent_node))
        appointment_builder.add_node("appointment_reply", traced("appointment_reply", reply_node(spec.intent)))
        appointment_builder.add_node("appointment_handoff", appointment_handoff)
        appointment_builder.add_edge(START, "appointment_subgraph_start")
        appointment_builder.add_edge("appointment_subgraph_start", "appointment_understand")
        appointment_builder.add_edge("appointment_understand", "appointment_tool_plan")
        appointment_builder.add_edge("appointment_tool_plan", spec.agent_name)
        appointment_builder.add_edge(spec.agent_name, "appointment_reply")
        appointment_builder.add_edge("appointment_reply", "appointment_handoff")
        appointment_builder.add_edge("appointment_handoff", END)
        return appointment_builder.compile()

    def build_points_subgraph(spec: AgentSpec, agent_node):
        """积分专属多轮子图：理解 → 只读/写工具 → 模型回复 → 等待或回交。"""
        def start(state: AgentState) -> dict:
            """标记当前领域子图开始处理本轮消息。"""
            return {"node_trace": [*state.get("node_trace", []), {
                "node": "points_subgraph_start", "status": "started", "duration_ms": 0,
            }]}

        points_builder = StateGraph(AgentState)
        points_builder.add_node("points_subgraph_start", start)
        points_builder.add_node("points_understand", traced("points_understand", points_understand))
        points_builder.add_node(spec.agent_name, traced(spec.agent_name, agent_node))
        points_builder.add_node("points_reply", traced("points_reply", reply_node(spec.intent)))
        points_builder.add_node("points_handoff", points_handoff)
        points_builder.add_edge(START, "points_subgraph_start")
        points_builder.add_edge("points_subgraph_start", "points_understand")
        points_builder.add_edge("points_understand", spec.agent_name)
        points_builder.add_edge(spec.agent_name, "points_reply")
        points_builder.add_edge("points_reply", "points_handoff")
        points_builder.add_edge("points_handoff", END)
        return points_builder.compile()

    def build_one_shot_subgraph(spec: AgentSpec, agent_node):
        """把单轮领域能力也封装成独立子图，并统一将结果交还父图。"""
        name = spec.agent_name
        def start(state: AgentState) -> dict:
            """标记当前领域子图开始处理本轮消息。"""
            trace_event("graph.node.start", summary={"node": f"{name}_subgraph"})
            return {
                "node_trace": [*state.get("node_trace", []), {
                    "node": f"{name}_subgraph", "status": "started", "duration_ms": 0,
                }],
            }

        def handoff(state: AgentState) -> dict:
            """记录子智能体完成后的控制权交接。"""
            status = state.get("result", {}).get("status", "completed")
            trace_event("agent.handoff", summary={
                "agent": spec.intent, "status": status, "control": "main",
            })
            return {
                "agent_status": "completed",
                "agent_handoff": {
                    "agent": spec.intent,
                    "status": status,
                    "control": "main",
                },
                "node_trace": [*state.get("node_trace", []), {
                    "node": f"{name}_handoff", "status": "completed", "duration_ms": 0,
                }],
            }

        subgraph_builder = StateGraph(AgentState)
        subgraph_builder.add_node(f"{name}_start", start)
        subgraph_builder.add_node(name, traced(name, agent_node))
        subgraph_builder.add_node(f"{name}_reply", traced(f"{name}_reply", reply_node(spec.intent)))
        subgraph_builder.add_node(f"{name}_handoff", handoff)
        subgraph_builder.add_edge(START, f"{name}_start")
        subgraph_builder.add_edge(f"{name}_start", name)
        subgraph_builder.add_edge(name, f"{name}_reply")
        subgraph_builder.add_edge(f"{name}_reply", f"{name}_handoff")
        subgraph_builder.add_edge(f"{name}_handoff", END)
        return subgraph_builder.compile()

    # 这里是在启动时“编译流程”，不是执行一次用户请求。每个子图也单独
    # compile()，再作为父图节点挂载；checkpoint 只放在父图，统一保存跨轮状态。
    # 节点通过同一 AgentState 传递消息、预约槽位和业务执行结果。
    builder = StateGraph(AgentState)
    builder.add_node("static_faq_cache", cache_lookup)
    builder.add_node("supervisor", supervisor)
    for spec in registry.specs():
        agent_node = spec.node_factory(dependencies)
        subgraph = (
            build_appointment_subgraph(spec, agent_node) if spec.workflow == "appointment"
            else build_points_subgraph(spec, agent_node) if spec.workflow == "points"
            else build_one_shot_subgraph(spec, agent_node)
        )
        builder.add_node(spec.subgraph_name, subgraph)
    builder.add_node("main_agent_response", main_agent_response)
    builder.add_node("general_reply", traced("general_reply", reply_node("general")))
    builder.add_node("main_agent_return", main_agent_return)
    builder.add_edge(START, "static_faq_cache")
    builder.add_conditional_edges(
        "static_faq_cache",
        lambda state: "hit" if state.get("cache_info", {}).get("status") == "hit" else "miss",
        {"hit": END, "miss": "supervisor"},
    )
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["intent"],
        {**registry.route_map(), "general": "main_agent_response"},
    )
    for spec in registry.specs():
        if spec.workflow in {"appointment", "points"}:
            builder.add_conditional_edges(
                spec.subgraph_name,
                lambda state: "return_to_main" if state.get("agent_status") == "completed" else "wait_for_user",
                {"return_to_main": "main_agent_return", "wait_for_user": END},
            )
        else:
            builder.add_edge(spec.subgraph_name, "main_agent_return")
    builder.add_edge("main_agent_response", "general_reply")
    builder.add_edge("general_reply", END)
    builder.add_edge("main_agent_return", END)
    # 传入 Checkpointer 后，相同 thread_id 可恢复之前的多轮状态；
    # 领域子图返回 END 仅结束本轮子图，不一定代表用户任务已完成。
    return builder.compile(checkpointer=checkpointer)


def build_conversational_graph(repo: Optional[AgentRepository] = None,
                               router: Optional[IntentRouter] = None):
    """文件会话库配 SQLite checkpoint，纯内存会话库使用 MemorySaver。"""
    # checkpointer 路径的 .env 配置必须在读取前加载；模型路由稍后也复用该环境。
    load_dotenv()
    repo = repo or AgentRepository()
    router = router or IntentRouter.from_env()
    legacy_api = _legacy_api_from_env(repo)
    mcp_queries = None
    transport = os.getenv("BLOOD_AGENT_BUSINESS_TRANSPORT", "http").strip().lower()
    # 旧礼品开关保留为过渡入口；启用后同样必须先启动独立业务 HTTP 服务。
    if transport == "mcp" or os.getenv("BLOOD_AGENT_MCP_REWARDS", "").lower() == "stdio":
        from blood_agents.business.mcp_client import MCPBusinessQueryClient

        mcp_queries = MCPBusinessQueryClient(
            os.getenv("BLOOD_AGENT_LEGACY_API_URL", "http://127.0.0.1:8001"),
            timeout=float(os.getenv("BLOOD_AGENT_LEGACY_API_TIMEOUT", "10")),
        )
    if repo.db_path == ":memory:":
        return ConversationGraph(build_graph(
            repo=repo, router=router, checkpointer=MemorySaver(),
            legacy_api=legacy_api, mcp_queries=mcp_queries,
        ), conversation_store=repo)

    # checkpoint 与业务表分文件，不能用脱敏对话历史重放图节点：那可能重放写工具。
    # thread_id 仍由 ConversationGraph 用 user_id + session_id 稳定哈希，重启后可恢复。
    business_path = Path(repo.db_path)
    checkpoint_path = Path(os.getenv("BLOOD_AGENT_CHECKPOINT_DB", "").strip() or
                           business_path.with_name(f"{business_path.stem}.checkpoints.db"))
    if str(checkpoint_path) == ":memory:" or checkpoint_path.resolve() == business_path.resolve():
        raise ValueError("checkpoint must use a separate persistent SQLite file")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saver = SQLAlchemySqliteSaver(checkpoint_path)
    try:
        graph = build_graph(
            repo=repo, router=router, checkpointer=saver,
            legacy_api=legacy_api, mcp_queries=mcp_queries,
        )
    except Exception:
        saver.close()
        raise
    return ConversationGraph(graph, conversation_store=repo, checkpoint_connection=saver)
