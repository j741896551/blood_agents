from blood_agents.ui.presentation import EMPTY_SUMMARY, execution_summary
from tests.test_conversation_tools import ask, setup


def test_summary_uses_runtime_nodes_and_clears_each_turn():
    graph, _, day = setup()
    first = ask(graph, f"预约 {day} 上午到市中心献血屋")
    assert [n["node"] for n in first["node_trace"]] == [
        "supervisor", "appointment_subgraph", "appointment_understand", "appointment_tool_plan",
        "appointment_agent",
        "appointment_reply", "appointment_handoff",
    ]
    text = execution_summary(first)
    assert "supervisor → appointment_subgraph → appointment_understand → appointment_tool_plan → appointment_agent → appointment_reply → appointment_handoff" in text
    assert "控制交接：appointment → 等待用户" in text
    assert "awaiting_confirmation" in text and "离线规则" in text
    assert "预约工具规划：rules / 未请求工具" in text
    assert "submit_appointment" not in text
    points = ask(graph, "查询积分")
    assert [n["node"] for n in points["node_trace"]] == [
        "supervisor", "points_subgraph_start", "points_understand", "points_agent", "points_reply", "points_handoff",
        "main_agent_return",
    ]
    assert "query_user_profile" not in execution_summary(points)


def test_summary_avoids_identity_and_handles_empty_error():
    assert execution_summary({}) == EMPTY_SUMMARY
    assert "本轮未完成" in execution_summary({"error": "missing_user_id"})
    text = execution_summary({"slots": {"user_id": "SECRET", "name": "SECRET"},
                              "result": {"data": {"mobile": "SECRET"}}})
    assert "SECRET" not in text
    assert "未记录" in text


def test_summary_explains_fallback_and_backup():
    text = execution_summary({"model_info": {"mode": "rules_fallback", "attempts": [
        {"provider": "primary", "status": "failed", "error_type": "timeout"},
    ]}})
    assert "规则降级" in text and "timeout" in text
    text = execution_summary({"model_info": {"mode": "llm", "provider": "backup"}})
    assert "实际模型服务：backup" in text
