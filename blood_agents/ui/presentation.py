"""将真实执行轨迹转成纯文本；不执行工具，也不推测未记录的节点。"""

import json

EMPTY_SUMMARY = "等待消息。本地 Mock 演示，不代表真实业务提交。"


def execution_summary(details: dict) -> str:
    """把节点和工具轨迹整理为页面可展示的执行摘要。"""
    if not details:
        return EMPTY_SUMMARY
    if details.get("error"):
        return "本轮未完成：" + str(details["error"])
    info = details.get("model_info", {})
    modes = {"rules": "离线规则", "llm": "模型理解", "rules_fallback": "模型失败 → 规则降级"}
    rows = ["本地 Mock 演示 · 本轮执行完成后生成的摘要（非实时进度）",
            "理解模式：" + modes.get(info.get("mode"), "未记录")]
    response = details.get("response_info", {})
    response_modes = {"llm": "模型生成", "risk_refusal": "风控拒答",
                      "unavailable": "模型未配置、不可用或回复/风控校验失败 → 服务不可用"}
    if response:
        rows.append("回复模式：" + response_modes.get(response.get("mode"), "未记录"))
        if response.get("provider"):
            rows.append("回复模型服务：" + response["provider"])
        if response.get("risk"):
            rows.append("回复风控：" + ("放行" if response["risk"].get("allow") else "拦截")
                        + f"（第 {response['risk'].get('attempt', '?')} 次）")
    plan = details.get("tool_plan_info", {})
    if plan:
        rows.append("预约工具规划：" + str(plan.get("mode", "未记录"))
                    + " / " + ("、".join(plan.get("requested_tools", [])) or "未请求工具"))
    if details.get("trace_id"):
        rows.append("本轮追踪 ID：" + str(details["trace_id"]))
    if info.get("provider"):
        rows.append("实际模型服务：" + info["provider"])
    for attempt in info.get("attempts", []):
        rows.append(f"模型尝试：{attempt['provider']} / {attempt['status']} / {attempt.get('error_type') or '-'}")
    nodes = details.get("node_trace", [])
    rows.append("节点流转：" + (" → ".join(n["node"] for n in nodes) or "未记录"))
    for node in nodes:
        rows.append(f"  {node['node']}：{node['status']}，{node['duration_ms']} ms")
    handoff = details.get("agent_handoff", {})
    if handoff:
        destination = "等待用户下一轮" if handoff.get("control") == "user" else "返回主 Agent"
        rows.append(f"控制交接：{handoff.get('agent', '领域 Agent')} → {destination}（{handoff.get('status', 'unknown')}）")
    rows.append("本轮意图：" + str(details.get("intent", "未记录")))
    result = details.get("result", {})
    status = result.get("status") or ("接口成功" if result.get("success") is True else
                                    "接口失败" if result.get("success") is False else "查询/处理完成")
    rows.append("处理状态：" + status)
    if result.get("code"):
        rows.append("接口返回码：" + str(result["code"]))
    # 只取预约选择，不复制实名资料、手机号、用户 ID 或完整接口返回。
    slots = details.get("slots", {})
    selected = {key: slots[key] for key in ("site", "date", "time_slot") if key in slots}
    if selected:
        rows.append("已收集参数：" + json.dumps(selected, ensure_ascii=False))
    trace = details.get("tool_trace", [])
    rows.append(f"工具调用：{len(trace)} 次")
    for index, tool in enumerate(trace, 1):
        arguments = {key: value for key, value in tool.get("arguments", {}).items()
                     if key in ("site", "date", "time_slot", "reward_id")}
        rows.append(f"  {index}. {tool['tool']} / {'写操作' if tool['mutating'] else '查询'} / "
                    f"{tool['status']} / {tool['duration_ms']} ms / "
                    f"{json.dumps(arguments, ensure_ascii=False)}")
    if not trace:
        rows.append("  本轮未调用业务工具。")
    return "\n".join(rows)
