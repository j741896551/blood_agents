"""会话标题模型只处理已保存的问答，且不能把首问原文当标题。"""

import json

from langchain_core.messages import AIMessage

from blood_agents.llm.conversation_summarizer import ConversationSummarizer


class DigestModel:
    def __init__(self, title, summary):
        self.title = title
        self.summary = summary
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=json.dumps({"title": self.title, "summary": self.summary},
                                            ensure_ascii=False))


def test_digest_uses_first_completed_exchange_without_business_invention():
    model = DigestModel("积分余额查询", "用户查询积分，助手回复可以帮助查询积分余额。")
    digest, info = ConversationSummarizer(model).summarize([
        {"user_message": "查询我的积分", "assistant_message": "我可以帮助查询积分余额。"},
    ])
    assert info["mode"] == "llm"
    assert digest.title == "积分余额查询"
    assert "查询我的积分" in model.calls[0][1].content
    assert "我可以帮助查询积分余额" in model.calls[0][1].content


def test_digest_rejects_copied_first_question_and_private_text():
    turns = [{"user_message": "查询我的积分", "assistant_message": "当前有演示积分。"}]
    copied, info = ConversationSummarizer(DigestModel(
        "查询我的积分", "用户查询积分，助手给出了当前积分答复。",
    )).summarize(turns)
    assert copied is None
    assert info["error_type"] == "title_copied_first_message"

    private, info = ConversationSummarizer(DigestModel(
        "积分查询进展", "用户手机号 13812345678，助手回复积分信息。",
    )).summarize(turns)
    assert private is None
    assert info["error_type"] == "ValueError"


def test_digest_without_model_is_unavailable():
    digest, info = ConversationSummarizer(None).summarize([
        {"user_message": "你好", "assistant_message": "您好"},
    ])
    assert digest is None
    assert info["reason"] == "model_not_configured"
