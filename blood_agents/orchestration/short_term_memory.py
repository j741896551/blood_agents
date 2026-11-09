"""从 LangGraph 已校验的流程状态提取短期任务记忆，不重读聊天文本猜事实。"""

from __future__ import annotations

from collections.abc import Mapping

from blood_agents.storage.privacy import sanitize_conversation_text

_APPOINTMENT_STATUSES = {"collecting", "awaiting_confirmation", "awaiting_action_confirmation"}
_APPOINTMENT_FIELDS = ("date", "site", "time_slot", "donor_kind", "donation_type", "donation_amount")
_POINTS_FIELDS = ("reward_id", "order_id", "address_id")


def _safe_fields(data: Mapping, names: tuple[str, ...]) -> dict:
    """槽位虽有白名单，仍要防止自由文本字段夹带常见个人资料。"""
    selected = {}
    for name in names:
        value = data.get(name)
        if value is None or value == "":
            continue
        if isinstance(value, str) and sanitize_conversation_text(value) != value:
            continue
        selected[name] = value
    return selected


def project_short_term_memory(state: Mapping) -> dict:
    """只投影当前任务和已收集参数；业务结果仍须本轮工具查询或回执证明。

    checkpoint 是流程状态的权威来源。此快照只辅助主导航理解“继续刚才的事”，
    不包含身份资料、积分余额、库存或操作成功结论，也不授权任何写操作。
    """
    active_agent = state.get("active_agent")
    appointment = state.get("appointment_state") or state.get("slots") or {}
    points = state.get("points_state") or {}
    tasks = []

    if appointment.get("status") in _APPOINTMENT_STATUSES:
        collected = _safe_fields(appointment, _APPOINTMENT_FIELDS)
        if appointment["status"] == "awaiting_action_confirmation":
            pending = "confirm_appointment_action"
        elif appointment["status"] == "awaiting_confirmation":
            pending = "confirm_appointment_preview"
        else:
            pending = "collect_appointment_details"
        tasks.append({"agent": "appointment", "status": appointment["status"],
                      "collected_params": collected, "pending_step": pending})

    if points.get("kind"):
        collected = _safe_fields(points, _POINTS_FIELDS)
        tasks.append({"agent": "points", "operation": points["kind"],
                      "collected_params": collected,
                      "pending_step": ("collect_address_details" if points["kind"] == "address_collect"
                                       else "await_user_confirmation")})

    current = next((task for task in tasks if task["agent"] == active_agent), None)
    suspended = [task for task in tasks if task is not current]
    return {"version": 1, "source": "agent_state", "active_agent": active_agent,
            "current_task": current, "suspended_tasks": suspended}
