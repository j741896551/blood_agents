"""OCR 子智能体：文件缺失时返回状态，不猜测图片内容。"""

from __future__ import annotations

import re

from blood_agents.core.models import AgentState

from .ocr_service import OCRService


def ocr_agent(ocr: OCRService, tools=None):
    """创建 OCR 领域执行节点。"""
    def node(state: AgentState) -> dict:
        """调用 OCR 服务并返回识别状态与文本。"""
        message = state["messages"][-1]
        text = message.content if hasattr(message, "content") else str(message)
        path_match = re.search(r"(?:图片|文件)[:：]?\s*([^\s]+)", text)
        if not path_match:
            return {"result": {"status": "need_image", "success": False,
                               "message": "本轮没有可识别的图片文件。"}}
        try:
            receipt = (tools.invoke("ocr_extract", image_path=path_match.group(1))
                       if tools else ocr.extract(path_match.group(1)))
        except (ValueError, RuntimeError, OSError):
            # 旧工具异常只说明本轮没得到 OCR 证据；路径和依赖异常原文不交给回复模型。
            return {"result": {"status": "ocr_failed", "success": False,
                               "message": "图片文字识别未成功，请检查图片后重试。"}}
        if not isinstance(receipt, dict) or not isinstance(receipt.get("text"), str) or not isinstance(
            receipt.get("engine"), str,
        ):
            return {"result": {"status": "ocr_failed", "success": False,
                               "message": "文字识别工具没有返回有效结果。"}}
        # 工具回执决定有无文字；模型只负责转述，不从图片文件名或用户描述猜内容。
        return {"result": {"status": "text_found" if receipt["text"].strip() else "no_text",
                           "success": True, "engine": receipt["engine"], "text": receipt["text"]}}
    return node
