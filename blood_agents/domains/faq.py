"""知识问答子智能体：先检索证据，回复模型不能自行补造业务事实。"""

from __future__ import annotations

from datetime import date

from blood_agents.core.models import AgentState
from blood_agents.domains.records import records_agent
from blood_agents.knowledge.donation_interval import (
    INTERVAL_RULE_SOURCE,
    DonationIntervalPlanner,
    calculate_next_donation_dates,
    is_personal_next_donation_question,
    normalize_donation_type,
)
from blood_agents.knowledge.evidence import EvidenceCheckUnavailable
from blood_agents.knowledge.rag import HybridRetriever
from blood_agents.log.trace_logging import trace_event
from blood_agents.storage.repository import AgentRepository


def faq_agent(repo: AgentRepository, retriever: HybridRetriever, tools=None,
              evidence_assessor=None, interval_planner: DonationIntervalPlanner | None = None):
    """创建使用知识检索与证据核验的 FAQ 节点。"""
    records_node = records_agent(repo, tools)

    def node(state: AgentState) -> dict:
        """检索知识证据并生成供回复节点使用的事实。"""
        message = state["messages"][-1]
        text = message.content if hasattr(message, "content") else str(message)
        try:
            hits = tools.invoke("knowledge_search", query=text) if tools else retriever.search(text)
        except Exception:  # noqa: BLE001 - 检索故障不是“知识库无答案”，不能让模型凭常识补齐
            trace_event("knowledge.search", summary={"status": "retrieval_failed", "hit_count": 0})
            return {"retrieved_context": [], "result": {
                "type": "rag", "status": "retrieval_failed", "citations": [],
                "message": "知识检索暂不可用。",
            }}
        if not isinstance(hits, list) or any(
            not isinstance(hit, dict) or not isinstance(hit.get("text"), str)
            or not hit["text"].strip() or not isinstance(hit.get("source"), str)
            or not hit["source"].strip() for hit in hits
        ):
            trace_event("knowledge.search", summary={"status": "retrieval_failed", "hit_count": 0})
            return {"retrieved_context": [], "result": {
                "type": "rag", "status": "retrieval_failed", "citations": [],
                "message": "知识检索结果格式异常。",
            }}
        # 相关片段不等于足以回答。只有模型确认支持问题所请求的具体事实，
        # 才把被指认的原始片段交给回复模型；核验失败不能退回“有命中即可信”。
        retrieval_hit_count = len(hits)
        evidence_info = {"mode": "not_configured"}
        if hits and evidence_assessor is not None:
            try:
                hits, evidence_info = evidence_assessor.assess(text, hits)
            except EvidenceCheckUnavailable as exc:
                trace_event("evidence.check", summary={
                    "mode": "unavailable", "error_type": exc.info.get("error_type"),
                    "candidate_count": retrieval_hit_count, "selected_count": 0,
                })
                return {"retrieved_context": [], "result": {
                    "type": "rag", "status": "evidence_check_failed", "citations": [],
                    "retrieval_hit_count": retrieval_hit_count,
                    "message": "证据核验暂不可用。",
                }}
        # 只有本轮核验保留下来的标准片段才可以触发个人间隔计算；
        # 通用问答仍只检索知识，不读取个人记录。
        interval_evidence = [hit for hit in hits if hit.get("source") == INTERVAL_RULE_SOURCE]
        interval_plan = None
        interval_plan_info = {"mode": "not_needed"}
        result = {
            "type": "rag",
            "status": "grounded" if hits else "no_evidence",
            "citations": list(dict.fromkeys(hit["source"] for hit in hits)),
            "retrieval_hit_count": retrieval_hit_count,
            "evidence_mode": evidence_info.get("mode"),
        }
        if interval_evidence and is_personal_next_donation_question(text) and interval_planner is not None:
            message = text
            interval_plan, interval_plan_info = interval_planner.plan(message, interval_evidence)
            result["interval_plan_info"] = interval_plan_info
            if interval_plan is None:
                result.update({"status": "interval_plan_unavailable",
                               "success": False,
                               "message": "献血间隔任务规划暂不可用，未查询个人献血记录。"})
            elif interval_plan.needs_personal_record:
                # 复用记录子 Agent 的身份白名单、接口校验及字段脱敏逻辑。
                records_result = records_node(state).get("result", {})
                if records_result.get("status") != "records_found":
                    interval_status = {
                        "no_records": "personal_interval_no_records",
                        "records_query_failed": "personal_interval_query_failed",
                    }.get(records_result.get("status"), "personal_interval_insufficient_history")
                    result.update({"status": interval_status,
                                   "success": records_result.get("success") is True,
                                   "personal_interval": {"status": interval_status}})
                else:
                    records = records_result.get("records") or []
                    try:
                        dated_records = [(date.fromisoformat(item["donated_at"]), item)
                                         for item in records]
                    except (TypeError, ValueError, KeyError):
                        dated_records = []
                    latest = max(dated_records, key=lambda item: item[0])[1] if dated_records else None
                    last_type = normalize_donation_type(latest.get("donation_type")) if latest else None
                    if latest is None or last_type is None:
                        result.update({"status": "personal_interval_insufficient_history",
                                       "success": True,
                                       "personal_interval": {
                                           "status": "personal_interval_insufficient_history",
                                           "reason": "latest_record_date_or_type_unknown",
                                       }})
                    else:
                        calculated = calculate_next_donation_dates(
                            latest["donated_at"], last_type,
                            interval_plan.target_donation_type,
                        )
                        if calculated is None:
                            result.update({"status": "personal_interval_insufficient_history",
                                           "success": True,
                                           "personal_interval": {
                                               "status": "personal_interval_insufficient_history",
                                               "reason": "date_or_type_unusable",
                                           }})
                        else:
                            result.update({"status": "personal_interval_calculated",
                                           "success": True,
                                           "personal_interval": calculated})
        # 检索阶段与个人业务事实一并交给回复节点； citations 仍绑定本轮核验来源。
        # 检索片段与来源必须同时存在；无命中、证据不足和核验失败分开交给回复模型。
        trace_event("knowledge.search", summary={
            "status": "grounded" if hits else "no_evidence", "hit_count": len(hits),
            "retrieval_hit_count": retrieval_hit_count,
            "evidence_mode": evidence_info.get("mode"),
            "backend": getattr(retriever, "backend", "unknown"),
            "sources": list(dict.fromkeys(hit["source"] for hit in hits)),
            "scores": [hit.get("score") for hit in hits],
            "rerank_scores": [hit.get("rerank_score") for hit in hits],
        })
        return {
            "retrieved_context": hits,
            "result": result,
        }
    return node
