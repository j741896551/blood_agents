"""只在测试中交付可确认的预约预览；其他回复保持模型不可用路径。"""

import json

from langchain_core.messages import AIMessage


class PreviewReplyModel:
    """让生命周期测试保留已展示预览的前提，不借规则模式越过确认门禁。"""

    def invoke(self, messages):
        facts = json.loads(messages[-1].content)
        status = facts.get("result_status")
        if status == "awaiting_confirmation":
            return AIMessage(content=(
                "这只是待确认预约，尚未提交。请核对本次献血人、网点和时间："
                + "、".join(facts["required_facts"])
                + "。如无误，请回复确认预约。"
            ))
        if status == "awaiting_action_confirmation":
            return AIMessage(content=(
                "这笔预约的操作尚未执行。请核对预约号并明确确认："
                + "、".join(facts["required_facts"])
            ))
        raise RuntimeError("测试模型只负责交付待确认预览")
