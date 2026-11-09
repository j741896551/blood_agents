"""业务工具注册与调用轨迹；写操作必须由领域流程满足确认边界后触发。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic

from blood_agents.log.trace_logging import trace_event

tool_trace = ContextVar("tool_trace", default=None)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., object]
    mutating: bool = False
    backend: str = "local"


class ToolRegistry:
    """统一登记本地/HTTP/MCP 工具并记录调用轨迹。

    注册不等于把所有工具直接交给模型。模型最多提出白名单只读工具名；
    写工具由领域节点在身份、参数和用户确认校验后调用。
    """

    def __init__(self) -> None:
        """初始化 ToolRegistry 的依赖与运行状态。"""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """登记一项能力并拒绝重复名称。"""
        if spec.name in self._tools:
            raise ValueError(f"重复工具名: {spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        """返回已注册工具名称列表。"""
        return sorted(self._tools)

    def invoke(self, name: str, **kwargs):
        """按注册名调用工具，并记录后端、耗时和有限的脱敏参数。

        此处只执行调用，不决定是否允许写入；身份、预览和确认条件由
        上层领域节点校验。异常继续抛出，避免把写入超时误说成失败回执。
        """
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        spec = self._tools[name]
        started = monotonic()
        # 白名单只记录业务选择，不记录用户资料、身份、文件路径或幂等键。
        arguments = {key: kwargs[key] for key in (
            "site", "date", "time_slot", "reward_id", "appointment_id", "redemption_id", "address_id",
        ) if key in kwargs}
        payload = kwargs.get("payload", {})
        arguments.update({key: payload[key] for key in ("site", "date", "time_slot") if key in payload})
        entry = {"tool": name, "mutating": spec.mutating,
                 "backend": spec.backend, "arguments": arguments}
        try:
            result = spec.handler(**kwargs)
            entry["status"] = "failed" if isinstance(result, dict) and result.get("success") is False else "success"
            if isinstance(result, dict):
                entry["code"] = result.get("code")
            return result
        except Exception as exc:  # record then propagate without retrying writes
            entry.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            entry["duration_ms"] = round((monotonic() - started) * 1000, 2)
            trace = tool_trace.get()
            if trace is not None:
                trace.append(entry)
            trace_event("tool.finished", summary={
                "tool": name, "mutating": spec.mutating, "backend": spec.backend,
                "status": entry["status"],
                "code": entry.get("code"), "duration_ms": entry["duration_ms"],
            }, full={"arguments": arguments})


def build_business_tools(repo, retriever=None, ocr=None, legacy_api=None,
                         mcp_queries=None) -> ToolRegistry:
    """按配置选择查询传输方式，向领域节点暴露稳定工具名。"""
    if legacy_api is None:
        raise ValueError("必须注入外部业务接口适配器，不能从 Agent 会话库读取业务数据")
    registry = ToolRegistry()
    legacy_backend = getattr(legacy_api, "transport_name", "external")
    query_points = (
        mcp_queries.get_points if mcp_queries else
        legacy_api.get_points
    )
    query_history = legacy_api.get_appointment_history
    query_appointments = legacy_api.get_user_appointments
    query_sites = legacy_api.get_sites_and_slots
    query_records = (
        mcp_queries.get_donation_records if mcp_queries else
        legacy_api.get_donation_records
    )
    submit_appointment = legacy_api.submit_appointment
    query_rewards = (
        mcp_queries.list_rewards if mcp_queries else
        legacy_api.list_rewards
    )
    redeem_reward = legacy_api.redeem_reward
    specs = [
        ToolSpec("query_user_profile", "查询已实名资料（身份来自入口）",
                 legacy_api.get_user_profile, backend=legacy_backend),
        ToolSpec("query_points", "通过旧业务接口查询用户积分", query_points,
                 backend="mcp_stdio" if mcp_queries else legacy_backend),
        ToolSpec("list_rewards", "查询可兑换礼品", query_rewards,
                 backend="mcp_stdio" if mcp_queries else legacy_backend),
        ToolSpec("redeem_reward", "原子扣减积分并兑换礼品", redeem_reward, True,
                 backend=legacy_backend),
        ToolSpec("query_redemption_orders", "查询当前用户兑换记录及订单状态",
                 legacy_api.get_redemption_orders, backend=legacy_backend),
        ToolSpec("cancel_redemption_order", "确认后取消当前用户未发货兑换订单",
                 legacy_api.cancel_redemption_order, True, legacy_backend),
        ToolSpec("query_shipping_addresses", "查询当前用户的收货地址簿",
                 legacy_api.get_shipping_addresses, backend=legacy_backend),
        ToolSpec("save_shipping_address", "确认后新增或修改收货地址",
                 legacy_api.save_shipping_address, True, legacy_backend),
        ToolSpec("set_default_shipping_address", "确认后设置默认收货地址",
                 legacy_api.set_default_shipping_address, True, legacy_backend),
        ToolSpec("set_order_shipping_address", "确认后填写订单收货信息",
                 legacy_api.set_order_shipping_address, True, legacy_backend),
        ToolSpec("query_records", "按入口用户身份查询献血记录", query_records,
                 backend="mcp_stdio" if mcp_queries else legacy_backend),
        ToolSpec("query_appointment_history", "查询用户历史预约与偏好", query_history,
                 backend=legacy_backend),
        ToolSpec("query_user_appointments", "按入口用户身份查询已提交预约", query_appointments,
                 backend=legacy_backend),
        ToolSpec("submit_appointment", "确认后调用旧业务预约接口", submit_appointment, True,
                 legacy_backend),
        ToolSpec("cancel_appointment", "确认后取消已提交预约",
                 legacy_api.cancel_appointment, True, legacy_backend),
        ToolSpec("checkin_appointment", "确认后为已预约记录签到",
                 legacy_api.checkin_appointment, True, legacy_backend),
        ToolSpec("query_sites", "查询网点配置及可预约时段", query_sites, backend=legacy_backend),
        ToolSpec("knowledge_search", "检索审核后的献血知识库", retriever.search),
        ToolSpec("ocr_extract", "识别证件或业务材料", ocr.extract),
    ]
    for spec in specs:
        registry.register(spec)
    return registry
