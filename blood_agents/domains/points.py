"""积分子 Agent：模型选动作，业务事实与写回执只取外部接口。"""

from __future__ import annotations

import re
import uuid

from blood_agents.business.legacy_api import LegacyBloodAPI
from blood_agents.core.models import AgentState
from blood_agents.storage.repository import AgentRepository

CONFIRM_WORDS = {
    "redeem": "确认兑换", "cancel_order": "确认取消订单",
    "address_add": "确认保存地址", "address_update": "确认保存地址",
    "address_default": "确认设为默认", "order_address": "确认填写收货信息",
}

READ_FAILURE_STATUS = {
    "query_points": "points_query_failed",
    "list_rewards": "reward_catalog_query_failed",
    "query_redemption_orders": "order_query_failed",
    "query_shipping_addresses": "address_query_failed",
}


def _mask_mobile(mobile: str) -> str:
    """只在对话预览中展示手机号首尾，完整值仅交给业务接口。"""
    return f"{mobile[:3]}****{mobile[-4:]}"


class BusinessReadUnavailable(RuntimeError):
    """只标识查询链路不可用；不能吞掉写接口可能已执行的异常。"""

    def __init__(self, tool_name: str) -> None:
        """初始化 BusinessReadUnavailable 的依赖与运行状态。"""
        super().__init__(tool_name)
        self.tool_name = tool_name


def points_agent(repo: AgentRepository, legacy_api: LegacyBloodAPI, tools=None):
    """跨轮只存待确认意向；无模型/预览失败时由交接节点清除意向。"""
    fallback = {
        "query_points": legacy_api.get_points, "list_rewards": legacy_api.list_rewards,
        "redeem_reward": legacy_api.redeem_reward,
        "query_redemption_orders": legacy_api.get_redemption_orders,
        "cancel_redemption_order": legacy_api.cancel_redemption_order,
        "query_shipping_addresses": legacy_api.get_shipping_addresses,
        "save_shipping_address": legacy_api.save_shipping_address,
        "set_default_shipping_address": legacy_api.set_default_shipping_address,
        "set_order_shipping_address": legacy_api.set_order_shipping_address,
    }

    def call(name, **kwargs):
        """调用指定业务工具并保留查询失败与写入异常的区别。"""
        try:
            return tools.invoke(name, **kwargs) if tools else fallback[name](**kwargs)
        except Exception as exc:  # 不把只读上游故障误报成空业务结果
            if name in READ_FAILURE_STATUS:
                raise BusinessReadUnavailable(name) from exc
            # 写请求失败时仍需保留“不确定是否执行”语义，不在这里生成成功/失败回执。
            raise

    def preview(kind, pending, message, **facts):
        """只生成待确认事实与服务端 pending，不扣积分、保存地址或取消订单。"""
        status = "awaiting_redemption_confirmation" if kind == "redeem" else "awaiting_points_confirmation"
        return {"points_state": pending, "result": {
            "status": status, "success": False, "operation": kind,
            "confirm_word": CONFIRM_WORDS[kind], "message": message, **facts,
        }}

    def redeem_preview(reward, balance, address, idempotency_key=None, *, address_saved=False):
        """完整订单预览包含礼品、成本和已查询的收货资料；只有这一步可产生确认态。"""
        pending = {"kind": "redeem", "reward_id": reward["reward_id"],
                   "reward": reward["name"], "points_cost": reward["points_cost"],
                   "points": balance, "address_id": address["address_id"],
                   "idempotency_key": idempotency_key or uuid.uuid4().hex}
        return preview("redeem", pending, "完整订单尚未提交，请核对礼品和收货信息后回复“确认兑换”。",
                       reward_id=reward["reward_id"], reward=reward["name"],
                       points_cost=reward["points_cost"], points=balance,
                       address=address, address_saved=address_saved)

    def node(state: AgentState) -> dict:
        """按积分意图读取接口事实；涉及写入的动作先生成/核对预览。

        points_state 保存的是上轮待确认操作，不是兑换结果；实际扣分、
        地址变更和订单取消均以对应写工具本轮返回的回执为准。
        """
        user_id = state["user_id"]  # 只取入口身份，不采信模型输出的身份。
        fields = state.get("understanding") or {}
        action = fields.get("action", "query")
        pending = dict(state.get("points_state") or {})
        utterance = next((msg.content for msg in reversed(state["messages"])
                          if getattr(msg, "type", None) == "human"), "")
        normalized = utterance.strip().strip("。！!?？")

        if action == "cancel":
            return {"points_state": {}, "result": {
                "status": "redemption_aborted" if pending else "no_pending_redemption",
                "success": bool(pending), "message": (
                    "已退出待确认操作，未调用写接口。" if pending else "目前没有待确认的积分或收货操作。"
                ),
            }}
        if pending.get("kind") in {"redeem", "redeem_address_selection"} and fields.get(
            "address_id"
        ) and action in {
            "query", "addresses", "redeem",
        }:
            # 用户在最终确认前改选地址，旧预览立即失效；重新读取当前用户的地址簿。
            response = call("query_shipping_addresses", user_id=user_id)
            if not response["success"]:
                return {"points_state": {}, "result": {**response, "status": "address_query_failed"}}
            address = next((item for item in response["data"]["addresses"]
                            if item["address_id"] == fields["address_id"]), None)
            if not address:
                return {"points_state": {}, "result": {"status": "address_selection_required",
                                                       "success": False,
                                                       "message": "未找到当前用户的该收货地址。"}}
            reward = {"reward_id": pending["reward_id"], "name": pending["reward"],
                      "points_cost": pending["points_cost"]}
            return redeem_preview(reward, pending["points"], address, pending.get("idempotency_key"))
        if action == "confirm":
            kind = pending.get("kind", "redeem" if "reward_id" in pending else "")
            if not kind:
                return {"points_state": {}, "result": {"status": "no_pending_redemption", "success": False,
                                                     "message": "目前没有待确认的操作。"}}
            if kind not in CONFIRM_WORDS:
                if kind == "redeem_address_selection":
                    return {"points_state": pending, "result": {
                        "status": "address_selection_required", "success": False,
                        "message": "请先明确选择本次兑换的收货地址，尚未下单。",
                    }}
                return {"points_state": pending, "result": {
                    "status": "address_details_required", "success": False,
                    "message": "收货信息尚未收集完整，不能确认保存。",
                }}
            info = state.get("model_info") or {}
            modes = ((info.get("navigation") or {}).get("mode"), (info.get("points") or {}).get("mode"))
            if normalized != CONFIRM_WORDS.get(kind) or modes != ("llm", "llm"):
                return preview(kind, pending, "尚未执行；请按预览中的确认词回复。",
                               reward_id=pending.get("reward_id"), reward=pending.get("reward"),
                               points_cost=pending.get("points_cost"), order_id=pending.get("order_id"),
                               address_id=pending.get("address_id"))
            if kind == "redeem":
                try:
                    receipt = call("redeem_reward", user_id=user_id, reward_id=pending["reward_id"],
                                   address_id=pending["address_id"],
                                   idempotency_key=pending["idempotency_key"])
                except ValueError as exc:
                    return {"points_state": {}, "result": {"status": "redemption_failed",
                                                         "success": False, "message": str(exc),
                                                         "reward_id": pending["reward_id"]}}
                return {"points_state": {}, "result": {"status": "redeemed", "success": True,
                                                     "reward_id": pending["reward_id"], **receipt}}
            tool = {"cancel_order": "cancel_redemption_order", "address_add": "save_shipping_address",
                    "address_update": "save_shipping_address",
                    "address_default": "set_default_shipping_address",
                    "order_address": "set_order_shipping_address"}[kind]
            args = {"user_id": user_id}
            if kind == "cancel_order":
                args["redemption_id"] = pending["order_id"]
            elif kind in {"address_add", "address_update"}:
                args.update(recipient=pending.get("recipient"), mobile=pending.get("mobile"),
                            detail=pending.get("address_detail"))
                if kind == "address_update":
                    args["address_id"] = pending["address_id"]
            elif kind == "address_default":
                args["address_id"] = pending["address_id"]
            else:
                args.update(redemption_id=pending["order_id"], address_id=pending["address_id"])
            receipt = call(tool, **args)
            if kind in {"address_add", "address_update"} and receipt.get("success") and pending.get(
                "resume_redeem"
            ):
                # 地址保存是独立写操作；保存成功后重新读取礼品/余额，只生成新订单预览，
                # 不能把“确认保存地址”当成“确认兑换”。
                try:
                    points_result = call("query_points", user_id=user_id)
                    if points_result.get("success"):
                        catalog = call("list_rewards")
                        catalog = [item.__dict__ if hasattr(item, "__dict__") else dict(item)
                                   for item in catalog]
                        reward = next((item for item in catalog
                                       if item["reward_id"] == pending["resume_redeem"]["reward_id"]), None)
                        if reward:
                            return redeem_preview(reward, points_result["data"]["points"], receipt["data"],
                                                  address_saved=True)
                except BusinessReadUnavailable:
                    # 地址写入已有成功回执；后续查询失败不能把同一确认重新留在待执行态。
                    return {"points_state": {}, "result": {
                        **receipt, "status": kind + "_completed", "operation": kind,
                        "resume_redemption_unavailable": True,
                        "message": "收货地址已保存；积分或礼品查询暂不可用，尚未提交兑换，请稍后重新发起。",
                    }}
            return {"points_state": {}, "result": {
                **receipt, "status": kind + ("_completed" if receipt.get("success") else "_failed"),
                "operation": kind,
            }}

        if action in {"orders", "order_status", "cancel_order", "order_address"}:
            response = call("query_redemption_orders", user_id=user_id)
            if not response["success"]:
                return {"points_state": pending, "result": {**response, "status": "order_query_failed"}}
            orders = response["data"]["orders"]
            order_id = fields.get("order_id") or next((o["redemption_id"] for o in orders
                                                        if o["redemption_id"] in utterance), None)
            selected = next((o for o in orders if o["redemption_id"] == order_id), None)
            if action in {"orders", "order_status"}:
                if order_id and not selected:
                    return {"points_state": pending, "result": {
                        "status": "order_not_found", "success": False,
                        "message": "未找到属于当前用户的兑换订单。",
                    }}
                visible = [selected] if selected else orders
                return {"points_state": pending, "result": {"status": "orders_found" if visible else "no_orders",
                                                           "success": True, "orders": visible}}
            if not selected:
                return {"points_state": {}, "result": {"status": "order_selection_required", "success": False,
                                                     "orders": orders, "message": "请明确提供要操作的兑换订单号。"}}
            if action == "cancel_order":
                if selected["status"] not in {"未支付", "待填写收货信息", "待发货"}:
                    return {"points_state": {}, "result": {"status": "order_cancel_unavailable",
                                                         "success": False, "order": selected,
                                                         "message": f"订单当前为{selected['status']}，不能取消。"}}
                return preview("cancel_order", {"kind": "cancel_order", "order_id": order_id},
                               "尚未取消订单，请核对后回复“确认取消订单”。", order=selected)
            addresses_response = call("query_shipping_addresses", user_id=user_id)
            if not addresses_response["success"]:
                return {"points_state": {}, "result": {**addresses_response,
                                                     "status": "address_query_failed"}}
            addresses = addresses_response["data"]["addresses"]
            address_id = fields.get("address_id") or next((a["address_id"] for a in addresses
                                                            if a["address_id"] in utterance), None)
            address = next((a for a in addresses if a["address_id"] == address_id), None)
            if not address:
                return {"points_state": {}, "result": {"status": "address_selection_required",
                                                     "success": False, "addresses": addresses,
                                                     "message": "请提供当前用户地址簿中的地址编号。"}}
            if selected["status"] not in {"待填写收货信息", "待发货"}:
                return {"points_state": {}, "result": {"status": "order_address_unavailable",
                                                     "success": False, "order": selected,
                                                     "message": f"订单当前为{selected['status']}，不能修改收货信息。"}}
            return preview("order_address", {"kind": "order_address", "order_id": order_id,
                                              "address_id": address_id},
                           "尚未修改订单收货信息，请核对后回复“确认填写收货信息”。",
                           order=selected, address=address)

        if action in {"addresses", "address_add", "address_update", "address_default"} or (
            pending.get("kind") == "address_collect" and action == "query"
        ):
            response = call("query_shipping_addresses", user_id=user_id)
            if not response["success"]:
                return {"points_state": {}, "result": {**response, "status": "address_query_failed"}}
            addresses = response["data"]["addresses"]
            if action == "addresses":
                address_id = fields.get("address_id") or next((a["address_id"] for a in addresses
                                                                if a["address_id"] in utterance), None)
                selected_address = next((a for a in addresses if a["address_id"] == address_id), None)
                if address_id and not selected_address:
                    return {"points_state": pending, "result": {
                        "status": "address_not_found", "success": False,
                        "message": "未找到属于当前用户的收货地址。",
                    }}
                visible_addresses = [selected_address] if selected_address else addresses
                return {"points_state": pending, "result": {"status": "addresses_found" if addresses else "no_addresses",
                                                           "success": True, "addresses": visible_addresses}}
            address_id = fields.get("address_id") or next((a["address_id"] for a in addresses
                                                            if a["address_id"] in utterance), None)
            selected = None
            if action in {"address_update", "address_default"}:
                selected = next((a for a in addresses if a["address_id"] == address_id), None)
                if not selected:
                    return {"points_state": {}, "result": {"status": "address_selection_required",
                                                         "success": False, "addresses": addresses,
                                                         "message": "请提供当前用户地址簿中的地址编号。"}}
                if action == "address_default":
                    return preview("address_default", {"kind": "address_default", "address_id": address_id},
                                   "尚未设置默认地址，请核对后回复“确认设为默认”。", address=selected)
            details = dict(pending) if pending.get("kind") == "address_collect" and action in {
                "query", "address_add"
            } else {"kind": "address_collect"}
            if pending.get("kind") in {"redeem", "redeem_address_selection"} and action == "address_add":
                # 用户修改的是未提交兑换的收货信息，不是历史订单。保留礼品意向；
                # 保存新地址后仍须重新查询余额/礼品并生成最终兑换预览。
                details["resume_redeem"] = {
                    "reward_id": pending["reward_id"], "reward": pending.get("reward"),
                }
            for key in ("recipient", "address_detail"):
                if fields.get(key):
                    details[key] = fields[key]
            mobile = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", utterance)
            if mobile:
                details["mobile"] = mobile.group(0)
            if action == "address_update":
                details["address_id"] = address_id
                if not any(details.get(key) for key in ("recipient", "mobile", "address_detail")):
                    return {"points_state": {}, "result": {
                        "status": "address_change_required", "success": False,
                        "message": "请明确要修改的收货人、手机号或详细地址。",
                    }}
            missing = [] if action == "address_update" else [
                key for key in ("recipient", "mobile", "address_detail") if not details.get(key)
            ]
            resume = details.get("resume_redeem") or {}
            if missing:
                return {"points_state": details, "result": {"status": "address_details_required",
                                                           "success": False, "missing": missing,
                                                           "reward": resume.get("reward"),
                                                           "pending_redemption": bool(resume),
                                                           "message": (
                                                               "本次兑换尚未提交；新收货信息尚未保存，"
                                                               "请补充缺失字段，手机号须为完整 11 位。"
                                                               if resume else
                                                               "请补充收货人、手机号和详细地址；尚未保存。"
                                                           )}}
            kind = "address_update" if details.get("address_id") else "address_add"
            details["kind"] = kind
            old = selected or {}
            display_address = {
                "address_id": details.get("address_id"),
                "recipient": details.get("recipient") or old.get("recipient"),
                "mobile": _mask_mobile(details["mobile"]) if details.get("mobile") else old.get("mobile"),
                "detail": details.get("address_detail") or old.get("detail"),
            }
            return preview(kind, details, (
                "本次兑换的新收货地址尚未保存；请先确认保存地址，随后还需确认兑换。"
                if resume else "尚未保存地址，请核对后回复“确认保存地址”。"
            ), address=display_address, reward=resume.get("reward"),
                pending_redemption=bool(resume))

        # 积分与目录由工具查询；推荐只是当前余额/库存的可兑换筛选。
        points_result = call("query_points", user_id=user_id)
        if not points_result["success"]:
            return {"points_state": pending, "result": {**points_result, "status": "points_query_failed"}}
        rewards = call("list_rewards")
        catalog = [item.__dict__ if hasattr(item, "__dict__") else dict(item) for item in rewards]
        balance = points_result["data"]["points"]
        if action == "recommend":
            eligible = [item for item in catalog if item["points_cost"] <= balance
                        and (item["stock"] is None or item["stock"] > 0)]
            return {"points_state": pending, "result": {"status": "recommendations_found" if eligible else
                                                           "no_recommendations", "success": True,
                                                           "points": balance, "rewards": eligible,
                                                           "message": "仅按积分与库存筛选可兑换礼品。"}}
        if action == "redeem":
            reward_id = fields.get("reward_id")
            # 理解节点不允许从历史对话/提示词里的目录猜编号。业务节点拿到本轮真实目录后，
            # 才把用户本轮明确说出的完整礼品名称映射为编号；多选或名称/编号冲突不能代选。
            named = [item for item in catalog if item["name"] and item["name"] in utterance]
            if len(named) > 1 or (reward_id and named and any(
                item["reward_id"] != reward_id for item in named
            )):
                return {"points_state": {}, "result": {
                    "status": "reward_selection_required", "success": False,
                    "message": "提到了多个礼品或礼品编号与名称不一致，请明确选择一项。",
                    "points": balance, "rewards": catalog,
                }}
            if not reward_id and len(named) == 1:
                reward_id = named[0]["reward_id"]
            selected = next((item for item in catalog if item["reward_id"] == reward_id), None)
            if selected is None:
                return {"points_state": {}, "result": {"status": "reward_not_found" if reward_id else
                                                     "reward_selection_required", "success": False,
                                                     "message": "礼品不存在，请从目录指定礼品编号。" if reward_id else
                                                     "请从礼品目录中指定要兑换的礼品编号。",
                                                     "points": balance, "rewards": catalog}}
            if selected["stock"] is not None and selected["stock"] <= 0:
                return {"points_state": {}, "result": {"status": "reward_unavailable", "success": False,
                                                     "message": "该礼品当前无库存。", "reward_id": reward_id,
                                                     "points": balance}}
            if balance < selected["points_cost"]:
                return {"points_state": {}, "result": {"status": "insufficient_points", "success": False,
                                                     "message": "当前积分不足，尚未兑换。", "reward_id": reward_id,
                                                     "points": balance, "points_cost": selected["points_cost"]}}
            response = call("query_shipping_addresses", user_id=user_id)
            if not response["success"]:
                return {"points_state": {}, "result": {**response, "status": "address_query_failed"}}
            addresses = response["data"]["addresses"]
            requested_id = fields.get("address_id")
            address = next((item for item in addresses if item["address_id"] == requested_id), None)
            if requested_id and not address:
                return {"points_state": {}, "result": {"status": "address_selection_required",
                                                       "success": False, "addresses": addresses,
                                                       "message": "未找到当前用户的该收货地址。"}}
            if not requested_id:
                address = next((item for item in addresses if item["is_default"]), None)
                if not address and len(addresses) == 1:
                    address = addresses[0]
            if address:
                return redeem_preview(selected, balance, address)
            if addresses:
                return {"points_state": {"kind": "redeem_address_selection", "reward_id": reward_id,
                                         "reward": selected["name"],
                                         "points_cost": selected["points_cost"], "points": balance},
                        "result": {"status": "address_selection_required", "success": False,
                                   "addresses": addresses, "reward": selected["name"],
                                   "points_cost": selected["points_cost"], "points": balance,
                                   "message": "请先选择本次兑换使用的收货地址。"}}
            # 没有地址时保留选中的礼品，但不建立可提交的兑换预览。
            details = {"kind": "address_collect", "resume_redeem": {"reward_id": reward_id}}
            for key in ("recipient", "address_detail"):
                if fields.get(key):
                    details[key] = fields[key]
            mobile = re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", utterance)
            if mobile:
                details["mobile"] = mobile.group(0)
            missing = [key for key in ("recipient", "mobile", "address_detail") if not details.get(key)]
            if not missing:
                details["kind"] = "address_add"
                display = {"recipient": details["recipient"],
                           "mobile": _mask_mobile(details["mobile"]),
                           "detail": details["address_detail"]}
                return preview("address_add", details,
                               "收货地址尚未保存，确认保存后还需另行确认完整兑换订单。",
                               address=display, reward_id=reward_id, reward=selected["name"],
                               points_cost=selected["points_cost"], points=balance)
            return {"points_state": details,
                    "result": {"status": "redemption_address_required", "success": False,
                               "reward_id": reward_id, "reward": selected["name"],
                               "points_cost": selected["points_cost"], "points": balance,
                               "missing": missing,
                               "message": "尚未下单；请先填写并确认收货人、手机号和详细地址。"}}
        return {"points_state": pending, "result": {"status": "points_found", "success": True,
                                                   "points": balance, "rewards": catalog}}
    def safe_node(state: AgentState) -> dict:
        """把积分节点查询失败转换为可追踪的结构化结果。"""
        try:
            return node(state)
        except BusinessReadUnavailable as exc:
            # 查询失败是失败事实，不是空积分/空订单；保留现有意向供用户稍后重试。
            return {"points_state": state.get("points_state") or {}, "result": {
                "status": READ_FAILURE_STATUS[exc.tool_name], "success": False,
                "code": "UPSTREAM_ERROR", "message": "业务查询暂不可用，请稍后重试。",
            }}

    return safe_node
