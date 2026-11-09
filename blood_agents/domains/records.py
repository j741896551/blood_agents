"""献血记录子智能体：只读查询入口用户的旧接口数据。"""

from __future__ import annotations

from blood_agents.core.models import AgentState


def records_agent(repo=None, tools=None):
    """创建按登录用户查询献血记录的执行节点。"""
    def node(state: AgentState) -> dict:
        """按入口用户查询献血记录，并把接口回执转成回复事实。"""
        # 身份只从入口上下文读取；用户话术和模型输出不能替换查询目标。
        user_id = state["user_id"]
        try:
            if tools is None:
                raise RuntimeError("献血记录查询工具未配置")
            receipt = tools.invoke("query_records", user_id=user_id)
        except Exception:  # noqa: BLE001 - 只读旧接口故障应形成失败事实，不能让模型猜记录
            return {"result": {"status": "records_query_failed", "success": False,
                               "code": "UPSTREAM_ERROR", "message": "献血记录查询暂不可用。"}}
        if not isinstance(receipt, dict) or receipt.get("success") is not True:
            return {"result": {"status": "records_query_failed", "success": False,
                               "code": receipt.get("code", "INVALID_RESPONSE")
                               if isinstance(receipt, dict) else "INVALID_RESPONSE",
                               "message": "献血记录查询未成功，请稍后重试。"}}
        # 第三方回执结构可能异常；仅把符合契约的列表当作查询结果。
        # 非字典 data 不能抛进图外，更不能被模型误解为空记录。
        data = receipt.get("data")
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, list):
            return {"result": {"status": "records_query_failed", "success": False,
                               "code": "INVALID_RESPONSE", "message": "献血记录查询结果格式异常。"}}
        public_records = []
        for item in records:
            if not isinstance(item, dict):
                break
            volume = item.get("volume_ml")
            amount = item.get("donation_amount", volume)
            unit = item.get("donation_unit") or ("mL" if isinstance(volume, int) else None)
            if (not isinstance(item.get("donated_at"), str)
                    or not isinstance(item.get("site"), str)
                    or not isinstance(item.get("points"), int)
                    or isinstance(item["points"], bool)
                    or not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0
                    or unit not in {"mL", "治疗量"}
                    or (unit == "mL" and (not isinstance(volume, int) or volume != amount))
                    or (unit == "治疗量" and volume is not None)):
                break
            optional = ("donation_id", "collection_org", "donation_type", "blood_type", "rhd_type")
            if any(item.get(key) is not None and not isinstance(item[key], str) for key in optional):
                break
            # 只暴露可展示的白名单，不把旧接口可能带回的姓名、证件号、手机号送给模型。
            public_records.append({
                "donation_id": item.get("donation_id"),
                "donated_at": item["donated_at"], "site": item["site"],
                "collection_org": item.get("collection_org"),
                "donation_type": item.get("donation_type"),
                "donation_amount": amount, "donation_unit": unit, "volume_ml": volume,
                "blood_type": item.get("blood_type"), "rhd_type": item.get("rhd_type"),
                "points": item["points"],
            })
        if len(public_records) != len(records):
            return {"result": {"status": "records_query_failed", "success": False,
                               "code": "INVALID_RESPONSE", "message": "献血记录查询结果格式异常。"}}
        return {"result": {"status": "records_found" if public_records else "no_records",
                           "success": True, "records": public_records,
                           "summary": {"donation_count": len(public_records),
                                       "total_volume_ml": sum(item["volume_ml"] or 0 for item in public_records),
                                       "whole_blood_count": sum(item["donation_type"] == "全血"
                                                                for item in public_records),
                                       "component_donation_count": sum(item["donation_unit"] == "治疗量"
                                                                       for item in public_records),
                                       "component_total_units": sum(item["donation_amount"]
                                                                    for item in public_records
                                                                    if item["donation_unit"] == "治疗量"),
                                       "earned_points": sum(item["points"] for item in public_records)}}}
    return node
