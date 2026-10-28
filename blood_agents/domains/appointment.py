"""预约子智能体的业务执行节点。

节点只管理预约状态和工具调用：先读取入口用户的资料/网点/历史，再收集槽位，
完整参数进入待确认；只有下一轮明确确认且时段仍可用，才提交外部业务接口。
用户可见答复在 llm/reply_composer.py，不能把这里的结构化 result 当成回复文案。
"""

from __future__ import annotations

import re
from typing import Optional

from blood_agents.business.legacy_api import LegacyBloodAPI, appointment_idempotency_key
from blood_agents.core.models import AgentState
from blood_agents.storage.repository import AgentRepository


def _last_text(state: AgentState) -> str:
    """从状态中提取最新一条用户消息文本。"""
    message = state["messages"][-1]
    return message.content if hasattr(message, "content") else str(message)


def _find_available_slot(sites: list[dict], slots: dict) -> Optional[dict]:
    """确保用户选择来自接口的可用时段，不能让模型自行编造。"""
    for site in sites:
        if site["site"] != slots.get("site"):
            continue
        for candidate in site["slots"]:
            if (
                candidate["date"] == slots.get("date")
                and candidate["time_slot"] == slots.get("time_slot")
                and candidate["remaining"] > 0
            ):
                return candidate
    return None


def _masked_mobile(mobile: str) -> str:
    """把预约资料中的手机号转换为掩码形式。"""
    return mobile[:3] + "****" + mobile[-4:]


def _donor_payload(slots: dict, profile: dict, history: list[dict]) -> dict | None:
    """登录用户是代办人；献血人可以是接口查到的本人，也可以由用户另行提供。"""
    if slots.get("donor_kind") == "other":
        name, mobile = slots.get("donor_name"), slots.get("donor_mobile")
        if not name or not mobile:
            return None
        return {"kind": "other", "name": name, "mobile": mobile,
                "mobile_masked": _masked_mobile(mobile), "gender": slots.get("donor_gender"),
                "id_card": slots.get("donor_id_card"), "blood_type": slots.get("donor_blood_type")}
    if slots.get("donor_kind") == "previous" and history and history[0].get("donor_name"):
        previous = history[0]
        return {"kind": "previous", "name": previous["donor_name"],
                "mobile_masked": previous["donor_mobile_masked"]}
    return {"kind": "self", "name": profile["name"],
            "mobile_masked": profile["mobile_masked"]}


def appointment_agent(repo: AgentRepository, legacy_api: LegacyBloodAPI, tools=None):
    """创建预约子图的执行节点；预约成败以旧业务接口回执为准。"""
    def node(state: AgentState) -> dict:
        """先处理查询/既有预览确认，再收集新参数并查询可约数据。

        slots 是跨轮意向，result 是本轮给回复模型的事实；只有已向用户
        展示的预览才允许下一轮写入，不能把模型的“确认”直接当作接口回执。
        """
        # state 来自 LangGraph checkpoint。复制槽位后修改，避免影响上一轮状态引用。
        user_id = state.get("user_id", "")
        slots = dict(state.get("slots", {}))
        original_slots = dict(slots)
        interpretation = state.get("understanding", {})
        action = interpretation.get("action", "collect")
        rejected_sites = interpretation.get("rejected_sites") or []
        prefetched = state.get("appointment_tool_data") or {}

        def read_tool(name: str, fallback):
            """优先使用模型请求的只读工具结果；缺关键事实时由确定性边界补查。"""
            if name in prefetched:
                return prefetched[name]
            return tools.invoke(name, user_id=user_id) if tools else fallback(user_id)

        if action == "query":
            # 查询是独立只读动作：保留尚未完成的预约槽位，不执行网点推荐或提交。
            response = read_tool("query_user_appointments", legacy_api.get_user_appointments)
            if not response["success"]:
                return {
                    "slots": slots,
                    "result": {**response, "status": "query_failed"},
                }
            appointments = response["data"]["appointments"]
            if not appointments:
                return {
                    "slots": slots,
                    "result": {"status": "no_appointments", "appointments": []},
                }
            return {
                "slots": slots,
                "result": {"status": "appointments_found", "appointments": appointments},
            }
        # 只有上一轮服务端预览锁定的预约号可被确认轮写入；不使用模型本轮猜测的目标。
        confirm_text = _last_text(state).strip().strip("。！!?？")
        expected_confirm = ("确认取消" if slots.get("pending_operation") == "cancel_appointment"
                            else "确认签到")
        if (slots.get("pending_operation") and slots.get("preview_delivered") and action == "confirm"
                and not interpretation.get("slots") and confirm_text in {"确认", expected_confirm}):
            operation = slots["pending_operation"]
            appointment_id = slots["pending_appointment_id"]
            receipt = (tools.invoke(operation, user_id=user_id, appointment_id=appointment_id)
                       if tools else getattr(legacy_api, operation)(user_id, appointment_id))
            clean = {key: value for key, value in slots.items()
                     if key not in {"pending_operation", "pending_appointment_id", "status",
                                    "preview_delivered"}}
            return {"slots": clean, "result": receipt}
        if slots.get("pending_operation"):
            if action in {"cancel", "checkin"} and interpretation.get("slots"):
                # 用户换了目标：旧确认预览失效，按本轮新目标重新只读筛选。
                for key in ("pending_operation", "pending_appointment_id", "status", "preview_delivered"):
                    slots.pop(key, None)
                original_slots = dict(slots)
            elif re.search(r"不(?:取消|签到|办)|算了|退出|放弃", confirm_text):
                clean = {key: value for key, value in slots.items()
                         if key not in {"pending_operation", "pending_appointment_id", "status",
                                        "preview_delivered"}}
                return {"slots": clean, "result": {"status": "action_aborted",
                        "message": "已退出待确认操作，已提交预约记录未改变。"}}
            else:
                response = read_tool("query_user_appointments", legacy_api.get_user_appointments)
                if not response["success"]:
                    return {"slots": slots, "result": {**response, "status": "query_failed"}}
                selected = next((item for item in response["data"]["appointments"]
                                 if item["appointment_id"] == slots["pending_appointment_id"]), None)
                if selected is None or selected["status"] != "booked":
                    return {"slots": {}, "result": {"status": "appointment_state_rejected",
                            "success": False, "code": "INVALID_STATUS",
                            "message": "待确认预约已不存在或状态已变化，请重新查询。"}}
                return {"slots": slots, "result": {"status": "awaiting_action_confirmation",
                        "operation": slots["pending_operation"], "appointment": selected,
                        "message": "尚未执行，请明确确认取消或签到。"}}
        # collect/confirm/cancel 三类动作先合并本轮明确参数；任何修改都会让旧预览失效。
        new_values = interpretation.get("slots", {})
        if new_values.get("donor_kind") and new_values["donor_kind"] != slots.get("donor_kind"):
            slots.pop("donor_name", None)
            slots.pop("donor_mobile", None)
        if slots.get("site") in rejected_sites:
            # 用户明确说无法前往旧网点时，旧预览立即失效；不能再等待其确认。
            slots.pop("site", None)
            slots.pop("slot_id", None)
            slots.pop("preview_delivered", None)
            slots["status"] = "collecting"
        # 修改参数后必须重新展示预览，即使同一条消息还说了“确认”。
        changed = any(slots.get(key) != value for key, value in new_values.items())
        if changed:
            slots["status"] = "collecting"
            slots.pop("slot_id", None)
            slots.pop("donor_confirmed", None)
            slots.pop("preview_delivered", None)
        # 日期变了但用户没重选时段，旧“下午”不能自动套到新日期上。
        # 由接口列出新日期可选时段，再让用户明确选择。
        if "date" in new_values and new_values["date"] != slots.get("date") and "time_slot" not in new_values:
            slots.pop("time_slot", None)
        slots.update(new_values)
        if rejected_sites:
            slots["status"] = "collecting"
            slots.pop("slot_id", None)
            slots.pop("preview_delivered", None)
        if action in {"cancel", "checkin"}:
            # “退出本次未提交意向”与“取消数据库里已预约记录”是两种操作。
            # 后者必须依赖已提交预约查询和接口回执，不能凭模型措辞伪称已取消。
            targeted_booking = bool(
                {"date", "site", "time_slot", "appointment_id"} & new_values.keys()
                or re.search(r"预约号|已预约|已提交|这(?:笔|条)预约|该(?:笔|条)预约|签到", _last_text(state))
            )
            if action == "cancel" and slots.get("flow") == "appointment" and not targeted_booking:
                return {
                    "slots": {},
                    "result": {"status": "cancelled", "scope": "draft",
                               "message": "仅退出当前未提交的预约意向，未修改已提交预约。"},
                }
            response = read_tool("query_user_appointments", legacy_api.get_user_appointments)
            if not response["success"]:
                return {"slots": original_slots, "result": {**response, "status": "query_failed"}}
            appointments = response["data"]["appointments"]
            selection = {key: new_values[key] for key in ("date", "site", "time_slot", "appointment_id")
                         if key in new_values}
            # 未查询网点配置时，仍可从该用户真实预约记录中识别明确说出的网点。
            mentioned_sites = {item["site"] for item in appointments if item["site"] in _last_text(state)}
            if len(mentioned_sites) == 1 and selection.get("site") not in mentioned_sites:
                selection["site"] = mentioned_sites.pop()
            matching = [item for item in appointments
                        if all(item["appointment_date" if key == "date" else key] == value
                               for key, value in selection.items())]
            operation = "cancel_appointment" if action == "cancel" else "checkin_appointment"
            if len(matching) != 1:
                return {"slots": original_slots, "result": {
                    "status": "appointment_selection_required", "operation": operation,
                    "appointments": matching if matching else appointments,
                    "message": "请明确指定唯一一条预约记录；本轮没有执行操作。"}}
            selected = matching[0]
            if selected["status"] != "booked":
                return {"slots": original_slots, "result": {
                    "status": "appointment_state_rejected", "operation": operation,
                    "success": False, "code": "INVALID_STATUS", "appointment": selected,
                    "message": f"当前预约状态为 {selected['status']}，不能再{'取消' if action == 'cancel' else '签到'}"}}
            pending = {**original_slots, "pending_operation": operation,
                       "pending_appointment_id": selected["appointment_id"],
                       "status": "awaiting_action_confirmation"}
            pending.pop("preview_delivered", None)
            return {"slots": pending, "result": {"status": "awaiting_action_confirmation",
                    "operation": operation, "appointment": selected,
                    "message": "尚未执行，请用户明确确认后再调用接口。"}}

        # user_id 来自登录上下文。先查询实名资料，不能从聊天文本中猜测身份。
        profile_result = read_tool("query_user_profile", legacy_api.get_user_profile)
        if not profile_result["success"]:
            return {
                "slots": {},
                "result": profile_result,
            }
        # 调用旧业务能力获取历史偏好和当前真实可预约配置。
        sites = read_tool("query_sites", legacy_api.get_sites_and_slots)
        history = read_tool("query_appointment_history", legacy_api.get_appointment_history)
        if not slots.get("flow"):
            slots["flow"] = "appointment"
            slots["status"] = "collecting"
            if history:
                slots["preferred_site"] = history[0]["site"]

        # “是/对”只确认刚展示的献血人，不等于授权写入；最后的“确认预约”
        # 则同时确认预览中的网点、时段与献血人，并在本轮直接调用旧接口。
        if slots.get("status") == "awaiting_confirmation" and _last_text(state).strip().strip("。！!?？") in {
            "是", "对", "没错", "是的", "就是他", "就是她",
        }:
            slots["donor_confirmed"] = True

        # 只有上一轮已生成预览，并且用户明确确认，才能执行写操作。
        if (slots.get("status") == "awaiting_confirmation" and slots.get("preview_delivered")
                and action == "confirm"):
            selected = _find_available_slot(sites, slots)
            if not selected:
                slots["status"] = "collecting"
                return {
                    "slots": slots,
                    "result": {"status": "slot_changed", "availability": sites},
                }
            # 实名资料由用户接口补齐，避免要求用户在对话中重复输入敏感信息。
            payload = {
                "user_id": user_id,
                "site": slots["site"],
                "date": slots["date"],
                "time_slot": slots["time_slot"],
                "slot_id": selected["slot_id"],
                "user_profile": profile_result["data"],
                "donor": _donor_payload(slots, profile_result["data"], history),
                "donation_type": slots.get("donation_type") or profile_result["data"].get("donation_type"),
                "donation_amount": slots.get("donation_amount") or profile_result["data"].get("donation_amount"),
            }
            # 幂等键保证接口重试不会生成两条相同预约。
            key = appointment_idempotency_key(user_id, slots)
            result = tools.invoke("submit_appointment", payload=payload, idempotency_key=key) if tools else legacy_api.submit_appointment(payload, key)
            if result["success"]:
                return {
                    "slots": {},
                    "result": result,
                }
            return {
                "slots": {},
                "result": result,
            }

        # 参数已由预约子图的理解节点校验；执行 Agent 不负责重做语义解析。
        missing = []
        if "date" not in slots:
            missing.append("日期（如 2026-10-01）")
        if "time_slot" not in slots:
            missing.append("时间段（上午/下午或具体时间）")
        if "site" not in slots:
            missing.append("献血点（如 市中心献血屋）")
        if slots.get("donor_kind") == "other":
            if not slots.get("donor_name"):
                missing.append("本次献血人姓名")
            if not slots.get("donor_mobile"):
                missing.append("本次献血人手机号")
            if not slots.get("donor_id_card"):
                missing.append("本次献血人身份证号")
            if not slots.get("donor_gender"):
                missing.append("本次献血人性别")
            if not slots.get("donor_blood_type"):
                missing.append("本次献血人血型")
            if not slots.get("donation_type"):
                missing.append("献血类型（全血/机采）")
            if not slots.get("donation_amount"):
                missing.append("献血量（全血 200/300/400cc，机采 1/2 单位）")
        if slots.get("donation_type") and slots.get("donation_amount"):
            allowed = {"全血": {200, 300, 400}, "机采": {1, 2}}
            if slots["donation_amount"] not in allowed.get(slots["donation_type"], set()):
                missing.append("与献血类型匹配的献血量")
        # 信息不足时将已获得参数保存在 slots，下一轮继续收集。
        if missing:
            available = [slot for site in sites if site["site"] not in rejected_sites
                         for slot in site["slots"]
                         if slot["remaining"] > 0
                         and (not slots.get("date") or slot["date"] == slots["date"])
                         and (not slots.get("site") or site["site"] == slots["site"])]
            if not available and (slots.get("date") or slots.get("site")):
                return {
                    "slots": slots,
                    "result": {"status": "slot_unavailable", "availability": sites,
                               "missing": missing, "history": history,
                               "excluded_sites": rejected_sites},
                }
            return {
                "slots": slots,
                "result": {
                    "status": "need_more_info",
                    "missing": missing,
                    "history": history,
                    "availability": sites,
                    "excluded_sites": rejected_sites,
                },
            }
        # 参数齐全后仍需和接口返回的可用时段进行严格匹配。
        selected = _find_available_slot(sites, slots)
        if not selected:
            slots["status"] = "collecting"
            return {
                "slots": slots,
                "result": {"status": "slot_unavailable", "availability": sites},
            }
        # 此处仅进入待确认状态，不会提交预约。
        slots["slot_id"] = selected["slot_id"]
        slots.setdefault("donor_kind", "previous" if history and history[0].get("donor_name") else "self")
        slots["status"] = "awaiting_confirmation"
        slots.pop("preview_delivered", None)
        slots.setdefault("donor_confirmed", False)
        donor = _donor_payload(slots, profile_result["data"], history)
        return {
            "slots": slots,
            "result": {
                "status": "awaiting_confirmation",
                "payload_preview": {"site": slots["site"], "date": slots["date"],
                                    "time_slot": slots["time_slot"], "slot_id": slots["slot_id"]},
                "donor_preview": {"kind": donor["kind"], "name": donor["name"],
                                  "mobile_masked": donor["mobile_masked"],
                                  "id_card_masked": (
                                      donor["id_card"][:3] + "***********" + donor["id_card"][-4:]
                                      if donor.get("id_card") else profile_result["data"].get("id_card_masked")
                                  ),
                                  "gender": donor.get("gender") or profile_result["data"].get("gender"),
                                  "blood_type": donor.get("blood_type") or profile_result["data"].get("blood_type")},
                "donation_type": slots.get("donation_type") or profile_result["data"].get("donation_type"),
                "donation_amount": slots.get("donation_amount") or profile_result["data"].get("donation_amount"),
                "donor_confirmed": slots["donor_confirmed"],
            },
        }
    return node
