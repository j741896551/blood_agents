from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

Intent = str


class AppointmentSlots(BaseModel):
    """模型只提取用户明确给出的预约与代约资料；登录身份仍由服务端管理。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    date: Optional[str] = None
    site: Optional[str] = Field(default=None, min_length=1, max_length=80)
    time_slot: Optional[str] = Field(default=None, min_length=1, max_length=20)
    donor_kind: Optional[Literal["previous", "self", "other"]] = None
    donor_name: Optional[str] = Field(default=None, min_length=2, max_length=40)
    donor_mobile: Optional[str] = Field(default=None, pattern=r"^1[3-9]\d{9}$")
    donor_id_card: Optional[str] = Field(default=None, pattern=r"^\d{17}[\dXx]$")
    donor_gender: Optional[Literal["男", "女"]] = None
    donor_blood_type: Optional[Literal["A", "B", "AB", "O"]] = None
    donation_type: Optional[Literal["全血", "机采"]] = None
    donation_amount: Optional[int] = None
    appointment_id: Optional[str] = Field(default=None, min_length=1, max_length=40)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        """校验日期字符串是否符合预约日期格式。"""
        if value is not None:
            from datetime import date

            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError("日期须为 YYYY-MM-DD")
        return value


class NavigationDecision(BaseModel):
    """主导航只决定领域；实际可选意图由 AgentRegistry 生成受限子类。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    intent: str


def navigation_decision_model(intents: tuple[str, ...]) -> type[NavigationDecision]:
    """把已注册领域转为 JSON Schema enum；未知意图在进入图前即被拒绝。"""
    return create_model(
        "NavigationDecision", __base__=NavigationDecision,
        intent=(Literal[tuple(intents)], ...),
    )


class AppointmentUnderstanding(BaseModel):
    """预约子智能体的语义契约；身份、业务状态及实际结果由服务端管理。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    slots: AppointmentSlots = Field(default_factory=AppointmentSlots)
    action: Literal["collect", "confirm", "cancel", "query", "checkin"] = "collect"


class PointsUnderstanding(BaseModel):
    """积分子智能体只提取本轮明确操作与引用 ID；业务状态由工具决定。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal[
        "query", "catalog", "recommend", "redeem", "confirm", "cancel",
        "orders", "order_status", "cancel_order", "addresses", "address_add",
        "address_update", "address_default", "order_address",
    ] = "query"
    reward_id: Optional[str] = Field(default=None, pattern=r"^R\d{3}$")
    order_id: Optional[str] = Field(default=None, min_length=3, max_length=40)
    address_id: Optional[str] = Field(default=None, min_length=3, max_length=40)
    recipient: Optional[str] = Field(default=None, min_length=2, max_length=40)
    address_detail: Optional[str] = Field(default=None, min_length=5, max_length=160)


class AgentState(TypedDict, total=False):
    """主图与子图共用的状态契约，不等同于数据库里的业务实体。

    每轮节点只返回需要更新的字段；messages 用 add_messages 累加，其他字段
    由新值覆盖。跨轮恢复依赖 LangGraph checkpoint，业务记录仍以接口查询为准。
    """
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    intent: Intent
    slots: dict[str, str]
    # 当前轮查询/写工具的结构化事实；回复模型只能据此表达业务结论。
    result: dict
    error: str
    route_confidence: float
    retrieved_context: list[dict]
    auth_token: str
    understanding: dict
    model_info: dict
    # 最终对用户措辞的来源，与导航/槽位理解的 model_info 分开记录。
    response_info: dict
    # 两种轨迹分别说明调用过哪些工具、走过哪些图节点，便于用 trace_id 排查。
    tool_trace: list[dict]
    node_trace: list[dict]
    # 主图据此决定下一轮续接哪个领域 Agent；预约与积分兑换均可跨轮等待。
    active_agent: Optional[str]
    agent_status: Optional[Literal["waiting", "completed"]]
    # 子 Agent 向主图交接的结构化结果，避免只靠自然语言判断是否结束。
    agent_handoff: dict
    # 预约子 Agent 的权威流程状态；slots 暂作兼容字段供既有 UI 与调用方展示。
    appointment_state: dict
    # 积分兑换的待确认状态与预约互不共享；写操作使用服务端生成的幂等键。
    points_state: dict
    # 从已校验流程状态投影的短期任务快照；只辅助导航，不作为业务事实或写入授权。
    short_term_memory: dict
    # 跨会话偏好来自用户明确原话和仓储事件，不代表实时业务状态或预约授权。
    long_term_preferences: list[dict]
    # 画像目前仅白名单保存模型归纳的稳定兴趣；仅在用户查询画像时提供给 general 回复。
    user_profile: dict
    profile_query: bool
    profile_status: str
    memory_event: dict
    # 仅完整命中经审核的静态标准问答时跳过模型；其余情况继续原图。
    cache_info: dict
    # 模型只能规划只读工具；执行结果和身份均由服务端管理。
    appointment_tool_data: dict
    tool_plan_info: dict


@dataclass(frozen=True)
class DonationRecord:
    donation_id: str
    donated_at: str
    site: str
    collection_org: Optional[str]
    donation_type: Optional[str]
    donation_amount: int
    donation_unit: str
    volume_ml: Optional[int]
    blood_type: Optional[str]
    rhd_type: Optional[str]
    points: int


@dataclass(frozen=True)
class Appointment:
    appointment_id: str
    user_id: str
    site: str
    appointment_date: str
    time_slot: str
    status: str = "booked"
    donor_name: str = ""
    donor_mobile_masked: str = ""
    booker_name: str = ""
    donor_gender: str = ""
    donor_id_card_masked: str = ""
    donor_blood_type: str = ""
    donation_type: str = ""
    donation_amount: int = 0
    donation_unit: str = ""


@dataclass(frozen=True)
class Reward:
    reward_id: str
    name: str
    points_cost: int
    stock: Optional[int]
