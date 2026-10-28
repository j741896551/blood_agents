"""OCR 事实边界：只有本轮识别工具的有效回执可进入用户答复。"""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from blood_agents.orchestration.graph import build_graph
from blood_agents.orchestration.router import IntentRouter
from local_business.repository import BloodRepository


class OCRNavigation:
    def invoke(self, _messages):
        return AIMessage(content='{"intent":"ocr"}')


class OCRReply:
    def invoke(self, messages):
        result = json.loads(messages[-1].content)["result"]
        status = result["status"]
        if status == "need_image":
            return AIMessage(content="请提供要识别的图片，本轮还没有收到图片文件。")
        if status == "ocr_failed":
            return AIMessage(content="图片文字识别未成功，请检查图片后重试。")
        if status == "no_text":
            return AIMessage(content="图片已处理，但没有识别到文字。")
        return AIMessage(content="识别到的原文如下：\n" + result["text"])


class OCRTool:
    def __init__(self, receipt=None, error=None):
        self.receipt = receipt
        self.error = error
        self.paths = []

    def extract(self, image_path):
        self.paths.append(image_path)
        if self.error:
            raise self.error
        return self.receipt


def ask(ocr, text, reply=None):
    repo = BloodRepository()
    repo.seed_demo_data()
    router = IntentRouter(llm=OCRNavigation(), response_llm=reply or OCRReply(), risk_llm=False)
    graph = build_graph(repo, router, ocr=ocr, checkpointer=MemorySaver())
    return graph.invoke(
        {"user_id": "U1001", "messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": "ocr-evidence"}},
    )


def test_ocr_no_image_does_not_call_tool_or_invent_text():
    ocr = OCRTool()
    state = ask(ocr, "请识别这张照片")
    assert state["result"]["status"] == "need_image"
    assert state["response_info"]["mode"] == "llm"
    assert ocr.paths == []
    assert state["tool_trace"] == []


def test_ocr_successful_empty_and_text_receipts_have_distinct_states():
    empty = ask(OCRTool({"engine": "fake", "text": ""}), "识别图片：demo.png")
    assert empty["result"]["status"] == "no_text"
    assert "没有识别到文字" in empty["messages"][-1].content
    found = ask(OCRTool({"engine": "fake", "text": "献血证\n血型 A 型"}), "识别图片：demo.png")
    assert found["result"]["status"] == "text_found"
    assert found["result"]["text"] == "献血证\n血型 A 型"
    assert "献血证\n血型 A 型" in found["messages"][-1].content
    assert found["response_info"]["mode"] == "llm"
    assert [item["tool"] for item in found["tool_trace"]] == ["ocr_extract"]


def test_ocr_exception_or_invalid_receipt_does_not_reach_reply_as_text():
    for ocr in (OCRTool(error=RuntimeError("private local path")), OCRTool(receipt=None)):
        state = ask(ocr, "识别图片：demo.png")
        assert state["result"]["status"] == "ocr_failed"
        assert "private local path" not in str(state["result"])
        assert state["response_info"]["mode"] == "llm"
        assert "未成功" in state["messages"][-1].content


def test_ocr_reply_cannot_claim_text_without_ocr_evidence():
    class InventedReply:
        def invoke(self, _messages):
            return AIMessage(content="已识别到图片中的姓名：张三。")

    for ocr, question in ((OCRTool(), "请识别这张照片"),
                          (OCRTool({"engine": "fake", "text": ""}), "识别图片：demo.png")):
        state = ask(ocr, question, InventedReply())
        assert state["response_info"]["mode"] == "unavailable"
        assert "张三" not in state["messages"][-1].content


def test_ocr_does_not_promise_unimplemented_browser_upload():
    class RevisingReply:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="请直接上传图片，我收到后会识别。")
            return AIMessage(content="本轮没有收到图片文件；请提供服务端可访问的图片文件路径。")

    model = RevisingReply()
    state = ask(OCRTool(), "请识别这张照片", model)
    assert model.calls == 2
    assert state["response_info"]["mode"] == "llm"
    assert "直接上传" not in state["messages"][-1].content


def test_ocr_reply_must_preserve_extracted_text():
    class ParaphraseReply:
        def invoke(self, _messages):
            return AIMessage(content="图片是一张献血证。")

    state = ask(OCRTool({"engine": "fake", "text": "原文：血型 A 型"}),
                "识别图片：demo.png", ParaphraseReply())
    assert state["result"]["status"] == "text_found"
    assert state["response_info"]["mode"] == "unavailable"
    assert "血型 A 型" not in state["messages"][-1].content
