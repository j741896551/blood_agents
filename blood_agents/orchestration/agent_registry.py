"""领域 Agent 的显式注册表：同一份定义供主导航、规则路由和主图接线使用。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AgentDependencies:
    """构建领域执行节点所需的依赖；不把身份或会话数据放在注册信息中。"""

    repo: object
    legacy_api: object
    retriever: object
    ocr: object
    tools: object
    evidence_assessor: object | None = None
    interval_planner: object | None = None


@dataclass(frozen=True)
class AgentSpec:
    """一次注册描述领域能力及其执行入口，而不是扫描业务代码猜测用途。"""

    intent: str
    description: str
    node_factory: Callable[[AgentDependencies], Callable]
    rule_match: Callable[[str], bool]
    workflow: Literal["one_shot", "appointment", "points"] = "one_shot"

    @property
    def subgraph_name(self) -> str:
        """返回该领域智能体对应的子图节点名称。"""
        return f"{self.intent}_subgraph"

    @property
    def agent_name(self) -> str:
        """返回该领域智能体在图中的执行节点名称。"""
        return f"{self.intent}_agent"


class AgentRegistry:
    """按注册顺序保存领域 Agent；顺序也是离线规则匹配的优先级。"""

    def __init__(self) -> None:
        """初始化 AgentRegistry 的依赖与运行状态。"""
        self._agents: dict[str, AgentSpec] = {}
        self._frozen = False

    def register(self, spec: AgentSpec) -> None:
        """登记一项能力并拒绝重复名称。"""
        if self._frozen:
            raise RuntimeError("图已编译，不能再注册 Agent；请重建图")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.intent) or spec.intent == "general":
            raise ValueError(f"无效或保留的 Agent 意图: {spec.intent}")
        if not spec.description.strip():
            raise ValueError(f"Agent 缺少能力描述: {spec.intent}")
        if not callable(spec.node_factory) or not callable(spec.rule_match):
            raise TypeError(f"Agent 缺少可调用的节点工厂或规则: {spec.intent}")
        if spec.workflow not in ("one_shot", "appointment", "points"):
            raise ValueError(f"未知 Agent 流程类型: {spec.workflow}")
        if spec.intent in self._agents:
            raise ValueError(f"重复 Agent 意图: {spec.intent}")
        if spec.workflow == "appointment" and spec.intent != "appointment":
            # 当前只有预约实现了专属多轮状态机；新增多轮领域不能误套此流程。
            raise ValueError("预约流程只能注册给 appointment；其他多轮流程需单独实现")
        if spec.workflow == "points" and spec.intent != "points":
            raise ValueError("积分流程只能注册给 points")
        self._agents[spec.intent] = spec

    def freeze(self) -> None:
        """主图编译前冻结，避免提示词/枚举在运行时与图节点脱节。"""
        if not self._agents:
            raise ValueError("至少需要注册一个领域 Agent")
        self._frozen = True

    def specs(self) -> tuple[AgentSpec, ...]:
        """返回当前注册的领域智能体配置。"""
        return tuple(self._agents.values())

    def intents(self) -> tuple[str, ...]:
        """列出当前注册的领域意图。"""
        return (*self._agents, "general")

    def navigation_catalog(self) -> str:
        """生成主导航提示词使用的领域能力清单。"""
        rows = [f"{spec.intent}={spec.description}" for spec in self.specs()]
        rows.append("general=问候、询问系统能力及不属于上述业务的问题")
        return "；".join(rows)

    def route_map(self) -> dict[str, str]:
        """生成意图到 LangGraph 子图名称的映射。"""
        return {spec.intent: spec.subgraph_name for spec in self.specs()}

    def rule_route(self, text: str) -> str:
        """用注册表中的规则匹配领域意图。"""
        normalized = text.lower()
        for spec in self.specs():
            if spec.rule_match(normalized):
                return spec.intent
        return "general"


def default_agent_registry() -> AgentRegistry:
    """已有五个领域的一次性声明；工具处理函数仍由各领域 Agent 决定。"""
    from blood_agents.domains import (
        appointment_agent,
        faq_agent,
        ocr_agent,
        points_agent,
        records_agent,
    )

    registry = AgentRegistry()
    registry.register(AgentSpec(
        "ocr", "图片、证件或业务材料的文字识别",
        lambda d: ocr_agent(d.ocr, d.tools),
        lambda text: any(word in text for word in ("识别图片", "ocr", "化验单", "身份证")),
    ))
    registry.register(AgentSpec(
        "appointment", "预约献血、查询预约记录、取消已提交预约、预约签到，以及补充或退出当前预约",
        lambda d: appointment_agent(d.repo, d.legacy_api, d.tools),
        lambda text: (
            any(word in text for word in ("预约", "签到", "时间段", "献血点", "献血屋", "献血车"))
            or bool(re.search(
                r"(?:想|要|准备|打算)(?:在|于)?(?:今天|明天|后天)?"
                r"(?:上午|下午|晚上)?(?:去|来)?(?:献血|捐血)", text,
            ))
        ),
        workflow="appointment",
    ))
    registry.register(AgentSpec(
        "points", "查询积分、推荐和兑换礼品，管理兑换订单与收货地址",
        lambda d: points_agent(d.repo, d.legacy_api, d.tools),
        lambda text: any(word in text for word in (
            "积分", "兑换", "礼品", "余额", "订单", "收货", "地址", "物流", "发货",
        )),
        workflow="points",
    ))
    registry.register(AgentSpec(
        "records", "查询个人献血记录与历史；个人下次献血日期问题交给 faq 组合处理",
        lambda d: records_agent(d.repo, d.tools),
        lambda text: any(word in text for word in ("记录", "历史", "献过", "献血量")),
    ))
    registry.register(AgentSpec(
        "faq", "回答献血知识；结合间隔规则与个人记录计算下次献血参考日期",
        lambda d: faq_agent(d.repo, d.retriever, d.tools, d.evidence_assessor, d.interval_planner),
        lambda text: any(word in text for word in ("献血", "捐血", "血液", "空腹")),
    ))
    return registry
