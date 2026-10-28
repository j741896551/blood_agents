"""从已核对的代码关系生成三页可编辑 draw.io 图及离线示意预览。

PNG 是使用 Pillow 按同一节点坐标绘制的示意预览；不是 draw.io 原生导出。
有 draw.io CLI 时可按 drawio-skill 的命令从 .drawio 重新导出精确 PNG。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"

COLORS = {
    "entry": ("#DAE8FC", "#6C8EBF"),
    "orchestration": ("#E1D5E7", "#9673A6"),
    "decision": ("#FFF2CC", "#D6B656"),
    "tool": ("#FFE6CC", "#D79B00"),
    "data": ("#D5E8D4", "#82B366"),
    "planned": ("#F5F5F5", "#8B939C"),
    "error": ("#F8CECC", "#B85450"),
}


def node(id_, label, x, y, w=240, h=80, role="entry", shape="box", source="", dashed=False,
         status=None):
    return {
        "id": id_, "label": label, "x": x, "y": y, "w": w, "h": h,
        "role": role, "shape": shape, "source": source, "dashed": dashed,
        "status": status or ("planned" if dashed else "code"),
    }


def edge(id_, source, target, label="", points=None, dashed=False, start=None, end=None):
    return {
        "id": id_, "source": source, "target": target, "label": label,
        "points": points or [], "dashed": dashed, "start": start, "end": end,
    }


target_architecture = {
    "id": "target-architecture",
    "name": "01 · 目标多智能体架构",
    "width": 1600,
    "height": 1250,
    "subtitle": "设计视图：主 Agent 导航；预约/积分子 Agent 跨轮管理；流程终态回交主 Agent（OCR 暂缓）",
    "nodes": [
        node("target-entry", "交互入口\nGradio / FastAPI / CLI", 630, 80, 340, 90,
             source="blood_agents/web.py; blood_agents/api.py; blood_agents/cli.py"),
        node("target-session", "会话上下文与 Checkpointer\n按 user_id + session_id 隔离\n保存 active_agent 与子图状态", 630, 230, 340, 120,
             "orchestration", source="blood_agents/orchestration/conversation.py; blood_agents/orchestration/graph.py"),
        node("target-main", "主 Agent / Supervisor\n识别意图并分派子图\n接收结果；下一轮继续导航", 630, 430, 340, 120,
             "orchestration", source="blood_agents/orchestration/graph.py"),
        node("target-appointment", "预约子 Agent / 子图\n收集信息 → 查历史与网点 → 预览\n确认后提交；明确拒绝/取消不提交\n预约状态跨轮保存", 120, 650, 400, 150,
             "tool", source="blood_agents/domains/appointment.py; blood_agents/orchestration/graph.py"),
        node("target-domain-agents", "并列的领域子 Agent / 子图\n积分：兑换与地址订单多轮流程\n记录：按登录用户查询献血历史\nFAQ：检索证据与模型核验", 1050, 650, 440, 150,
             "tool", source="blood_agents/domains/points.py; blood_agents/domains/records.py; blood_agents/domains/faq.py"),
        node("target-terminal", "预约流程已到终态？", 210, 890, 220, 110,
             "decision", "diamond", "blood_agents/core/models.py"),
        node("target-wait", "尚未结束：等待用户下一轮\n保留 active_agent = appointment", 120, 1070, 400, 100,
             "entry", source="blood_agents/orchestration/conversation.py"),
        node("target-return", "子图回传 agent_handoff\n完成 / 用户明确拒绝或取消\n控制权交回主 Agent", 630, 900, 340, 120,
             "data", source="blood_agents/core/models.py; blood_agents/orchestration/graph.py"),
    ],
    "edges": [
        edge("t01", "target-entry", "target-session", "身份 + 会话"),
        edge("t02", "target-session", "target-main", "首次导航 / active_agent 续接"),
        edge("t03", "target-main", "target-appointment", "识别为预约", start="bottom", end="top"),
        edge("t04", "target-main", "target-domain-agents", "识别为其他领域任务", start="bottom", end="top"),
        edge("t05", "target-appointment", "target-terminal", "预约流程状态", start="bottom", end="top"),
        edge("t06", "target-terminal", "target-wait", "否：继续收集 / 等确认"),
        edge("t07", "target-wait", "target-session", "同会话恢复并续接", start="left", end="left",
             points=[(40, 1120), (40, 290), (590, 290)]),
        edge("t08", "target-terminal", "target-return", "是：完成 / 拒绝 / 取消",
             start="right", end="left"),
        edge("t09", "target-domain-agents", "target-return", "完成后回传领域结果",
             start="bottom", end="right"),
        edge("t10", "target-return", "target-main", "下一轮重新导航", start="right", end="right",
             points=[(1530, 960), (1530, 490)]),
    ],
}


architecture = {
    "id": "architecture",
    "name": "02 · 当前代码架构",
    "width": 1850,
    "height": 1130,
    "subtitle": "当前代码调用：平台 API 主备、SQLite 会话状态、可选 BGE 向量检索；灰框仅说明暂缓/范围外",
    "nodes": [
        node("entry", "交互入口\nGradio / FastAPI / CLI", 700, 110, 340, 90,
             source="blood_agents/web.py; blood_agents/api.py; blood_agents/cli.py"),
        node("conversation", "会话隔离 + 缓存前置\nuser_id + session_id\n缓存未命中才进入 Supervisor", 700, 250, 340, 110,
             "orchestration", source="blood_agents/orchestration/conversation.py; blood_agents/orchestration/graph.py; blood_agents/knowledge/static_faq_cache.py"),
        node("memory", "SQLite Checkpointer\n文件库持久化\n内存库 MemorySaver", 1220, 250, 300, 110,
             "data", "cylinder", "blood_agents/orchestration/graph.py"),
        node("supervisor", "LangGraph Supervisor\n领域意图路由", 700, 410, 340, 100,
             "orchestration", source="blood_agents/orchestration/graph.py"),
        node("router", "IntentRouter\n导航/预约分层理解 + 规则降级", 130, 410, 340, 100,
             "orchestration", source="blood_agents/orchestration/router.py"),
        node("gateway", "ModelGateway\n主备切换 / 超时 / 熔断", 130, 580, 340, 90,
             "orchestration", source="blood_agents/llm/model_gateway.py"),
        node("providers", "第三方平台主/备模型 API\nOpenAI 兼容；非本地生成", 50, 740, 330, 90,
             "data", source="blood_agents/orchestration/router.py; blood_agents/llm/model_gateway.py"),
        node("rules", "离线规则\n无密钥或模型失败时使用", 420, 740, 300, 90,
             "decision", source="blood_agents/orchestration/router.py"),
        node("agents", "五个编译领域子图\n预约/积分：独立理解与跨轮状态\n记录/FAQ：单轮；OCR 原型暂缓", 1170, 410, 470, 120,
             "orchestration", source="blood_agents/orchestration/graph.py; blood_agents/domains/appointment.py"),
        node("main_return", "main_agent_return\n接收 agent_handoff 后返回入口", 700, 590, 340, 80,
             "orchestration", source="blood_agents/orchestration/graph.py"),
        node("tools", "ToolRegistry\n业务调用与脱敏执行轨迹", 1170, 600, 470, 90,
             "tool", source="blood_agents/business/tools.py"),
        node("legacy", "MockLegacyBloodAPI\n资料 / 历史 / 网点 / 预约 / 积分", 940, 770, 360, 110,
             "tool", source="blood_agents/business/legacy_api.py"),
        node("ragocr", "FAQ 知识检索\n默认关键词\nBGE+FAISS / Milvus Lite（可选）\n可选 bge-reranker 精排\n模型证据核验", 1350, 770, 360, 150,
             "tool", source="blood_agents/knowledge/rag.py; blood_agents/knowledge/adapters.py"),
        node("sqlite", "SQLite 演示库\n用户 / 积分 / 记录 / 新预约", 940, 960, 360, 90,
             "data", "cylinder", "blood_agents/business/repository.py"),
        node("future", "范围说明：OCR / LoRA 暂缓\nvLLM / TGI 非当前范围\nMCP stdio 仅礼品目录只读工具", 1350, 960, 360, 90,
             "planned", source="blood_agents/business/mcp_server.py; deploy/docker-compose.yml",
             status="scope-note"),
    ],
    "edges": [
        edge("a01", "entry", "conversation", "消息 + 身份/会话"),
        edge("a02", "conversation", "memory", "checkpoint"),
        edge("a03", "conversation", "supervisor", "静态缓存未命中"),
        edge("a04", "supervisor", "router", "navigate", start="left", end="right"),
        edge("a05", "router", "gateway", "配置模型时"),
        edge("a06", "gateway", "providers", "LLM 调用"),
        edge("a07", "router", "rules", "", start="right", end="top",
             points=[(540, 460), (540, 700), (570, 700)]),
        edge("a08", "supervisor", "agents", "条件路由", start="right", end="left"),
        edge("a09", "agents", "tools", "调用工具"),
        edge("a13", "agents", "main_return", "单轮完成 / 预约终态", start="left", end="right"),
        edge("a10", "tools", "legacy", "旧系统契约", start="bottom", end="top",
             points=[(1400, 730), (1120, 730)]),
        edge("a11", "tools", "ragocr", "知识检索", start="bottom", end="top",
             points=[(1400, 730), (1530, 730)]),
        edge("a12", "legacy", "sqlite", "演示数据"),
    ],
}


appointment_flow = {
    "id": "appointment-flow",
    "name": "03 · 当前预约子图流程",
    "width": 1940,
    "height": 1600,
    "subtitle": "每轮由主图恢复并派发；预约未终结时等待用户，终态返回主图；提交以 Mock 接口返回为准",
    "nodes": [
        node("message", "用户发来本轮消息", 690, 80, 330, 70, "entry", "ellipse",
             "blood_agents/web.py"),
        node("restore", "按用户 + 会话恢复预约子图状态", 690, 190, 330, 70, "orchestration",
             source="blood_agents/orchestration/conversation.py"),
        node("understand", "主图只识别预约意图并派发\n子图 appointment_understand 提取槽位/动作", 690, 300, 330, 80,
             "orchestration", source="blood_agents/orchestration/graph.py; blood_agents/orchestration/router.py"),
        node("cancel", "取消？", 760, 430, 190, 100, "decision", "diamond",
             "blood_agents/domains/appointment.py"),
        node("cancelled", "清空预约状态并退出\n未提交，交回主图", 1320, 440, 320, 80, "error",
             source="blood_agents/domains/appointment.py"),
        node("profile", "查询已实名资料", 690, 570, 330, 70, "tool",
             source="blood_agents/domains/appointment.py; blood_agents/business/legacy_api.py"),
        node("has_profile", "资料可用？", 760, 690, 190, 100, "decision", "diamond",
             "blood_agents/domains/appointment.py"),
        node("profile_failed", "返回实名认证失败\n清空状态并交回主图", 1320, 700, 320, 80, "error",
             source="blood_agents/domains/appointment.py"),
        node("site_history", "查询网点时段 + 预约历史\n合并本轮槽位；仅提示历史偏好", 690, 830, 330, 90,
             "tool", source="blood_agents/domains/appointment.py"),
        node("confirmed", "上一轮已预览\n且本轮明确确认？", 685, 945, 340, 120,
             "decision", "diamond", "blood_agents/domains/appointment.py"),
        node("recheck", "重新校验可用时段\n调用 submit_appointment", 1320, 960, 320, 100,
             "tool", source="blood_agents/domains/appointment.py; blood_agents/business/legacy_api.py"),
        node("submit_result", "根据接口返回码回答\n成功/失败均清状态并回主图", 1320, 1110, 320, 90,
             "data", source="blood_agents/domains/appointment.py"),
        node("missing", "日期 / 网点 / 时段齐全？", 690, 1120, 330, 90,
             "decision", "diamond", "blood_agents/domains/appointment.py"),
        node("ask_missing", "追问缺少的参数\n保存 collecting 状态", 200, 1120, 320, 90,
             "entry", source="blood_agents/domains/appointment.py"),
        node("available", "组合在可用配置内？", 690, 1270, 330, 90,
             "decision", "diamond", "blood_agents/domains/appointment.py"),
        node("ask_reselect", "返回可选网点/时段\n请用户重新选择", 200, 1270, 320, 90,
             "entry", source="blood_agents/domains/appointment.py"),
        node("preview", "生成预约预览\n保存 awaiting_confirmation\n等待下一轮确认", 690, 1430, 330, 100,
             "data", source="blood_agents/domains/appointment.py"),
        node("loop", "下一轮消息继续此会话", 1320, 1430, 320, 80,
             "entry", source="blood_agents/orchestration/conversation.py"),
    ],
    "edges": [
        edge("f01", "message", "restore"),
        edge("f02", "restore", "understand"),
        edge("f03", "understand", "cancel"),
        edge("f04", "cancel", "cancelled", "是", start="right", end="left"),
        edge("f05", "cancel", "profile", "否"),
        edge("f06", "profile", "has_profile"),
        edge("f07", "has_profile", "profile_failed", "否", start="right", end="left"),
        edge("f08", "has_profile", "site_history", "是"),
        edge("f09", "site_history", "confirmed"),
        edge("f10", "confirmed", "recheck", "是", start="right", end="left"),
        edge("f11", "recheck", "submit_result", "接口返回"),
        edge("f12", "confirmed", "missing", "否"),
        edge("f13", "missing", "ask_missing", "否", start="left", end="right"),
        edge("f14", "missing", "available", "是"),
        edge("f15", "available", "ask_reselect", "否", start="left", end="right"),
        edge("f16", "available", "preview", "是"),
        edge("f17", "preview", "loop", "等明确确认", start="right", end="left"),
        edge("f18", "ask_missing", "loop", "", start="right", end="left",
             points=[(580, 1165), (580, 1380), (1290, 1380), (1290, 1470)]),
        edge("f19", "ask_reselect", "loop", "", start="bottom", end="bottom",
             points=[(360, 1550), (1480, 1550)]),
    ],
}


def drawio_page(page):
    diagram = ET.Element("diagram", {"id": page["id"], "name": page["name"]})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1200", "dy": "850", "grid": "1", "gridSize": "10", "page": "1",
        "pageScale": "1", "pageWidth": str(page["width"]),
        "pageHeight": str(page["height"]), "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    for item in page["nodes"]:
        fill, stroke = COLORS[item["role"]]
        shape = {"box": "rounded=1;arcSize=10", "ellipse": "ellipse",
                 "diamond": "rhombus", "cylinder": "shape=cylinder3"}[item["shape"]]
        style = (f"{shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 "fontColor=#1C3040;fontSize=17;align=center;verticalAlign=middle;"
                 "strokeWidth=2;spacing=8;")
        if item["dashed"]:
            style += "dashed=1;dashPattern=6 4;"
        cell = ET.SubElement(root, "mxCell", {
            "id": item["id"], "value": item["label"], "style": style,
            "vertex": "1", "parent": "1", "data-model-id": item["id"],
            "data-source": item["source"], "data-status": item["status"],
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(item["x"]), "y": str(item["y"]),
            "width": str(item["w"]), "height": str(item["h"]), "as": "geometry",
        })
    for link in page["edges"]:
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                 "html=1;strokeColor=#647887;strokeWidth=2;endArrow=block;endSize=8;"
                 "fontSize=14;fontColor=#324555;labelBackgroundColor=#FFFFFF;")
        if link["dashed"]:
            style += "dashed=1;"
        pins = {"top": (0.5, 0), "bottom": (0.5, 1),
                "left": (0, 0.5), "right": (1, 0.5)}
        for prefix, side in (("exit", link["start"]), ("entry", link["end"])):
            if side:
                px, py = pins[side]
                style += f"{prefix}X={px};{prefix}Y={py};{prefix}Dx=0;{prefix}Dy=0;"
        cell = ET.SubElement(root, "mxCell", {
            "id": link["id"], "value": link["label"], "style": style,
            "edge": "1", "parent": "1", "source": link["source"],
            "target": link["target"], "data-model-id": link["id"],
        })
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if link["points"]:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for x, y in link["points"]:
                ET.SubElement(array, "mxPoint", {"x": str(x), "y": str(y)})
    return diagram


def side_point(item, side):
    cx, cy = item["x"] + item["w"] / 2, item["y"] + item["h"] / 2
    return {"top": (cx, item["y"]), "bottom": (cx, item["y"] + item["h"]),
            "left": (item["x"], cy), "right": (item["x"] + item["w"], cy)}[side]


def arrow(draw, points, color="#647887"):
    draw.line(points, fill=color, width=3, joint="curve")
    if len(points) < 2:
        return
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 14
    a = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    b = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    draw.polygon([(x2, y2), a, b], fill=color)


def preview(page, destination):
    """无需 draw.io 桌面程序的示意预览；文字/配色与可编辑源一致。"""
    image = Image.new("RGB", (page["width"], page["height"]), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(FONT, 28)
    body = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 15)
    draw.text((50, 22), page["name"], font=title, fill="#1C3040")
    draw.text((50, 62), page["subtitle"], font=small, fill="#647887")
    by_id = {n["id"]: n for n in page["nodes"]}
    for link in page["edges"]:
        source, target = by_id[link["source"]], by_id[link["target"]]
        dx = (target["x"] + target["w"] / 2) - (source["x"] + source["w"] / 2)
        dy = (target["y"] + target["h"] / 2) - (source["y"] + source["h"] / 2)
        start_side = link["start"] or ("bottom" if dy > 0 and abs(dy) >= abs(dx) else
                                        "top" if dy < 0 and abs(dy) >= abs(dx) else
                                        "right" if dx > 0 else "left")
        end_side = link["end"] or ("top" if start_side == "bottom" else
                                  "bottom" if start_side == "top" else
                                  "left" if start_side == "right" else "right")
        first = side_point(source, start_side)
        last = side_point(target, end_side)
        pts = [first, *link["points"], last]
        if len(pts) == 2 and first[0] != last[0] and first[1] != last[1]:
            if start_side in {"top", "bottom"}:
                midpoint = (first[1] + last[1]) / 2
                pts = [first, (first[0], midpoint), (last[0], midpoint), last]
            else:
                midpoint = (first[0] + last[0]) / 2
                pts = [first, (midpoint, first[1]), (midpoint, last[1]), last]
        arrow(draw, pts)
        if link["label"]:
            midx = (first[0] + last[0]) / 2
            midy = (first[1] + last[1]) / 2
            bbox = draw.textbbox((0, 0), link["label"], font=small)
            width, height = bbox[2], bbox[3]
            draw.rectangle((midx - width / 2 - 5, midy - height / 2 - 4,
                            midx + width / 2 + 5, midy + height / 2 + 4), fill="white")
            draw.text((midx - width / 2, midy - height / 2), link["label"], font=small,
                      fill="#324555")
    for item in page["nodes"]:
        fill, stroke = COLORS[item["role"]]
        x, y, w, h = (item[k] for k in ("x", "y", "w", "h"))
        rect = (x, y, x + w, y + h)
        if item["shape"] == "ellipse":
            draw.ellipse(rect, fill=fill, outline=stroke, width=3)
        elif item["shape"] == "diamond":
            draw.polygon([(x + w / 2, y), (x + w, y + h / 2),
                          (x + w / 2, y + h), (x, y + h / 2)], fill=fill)
            draw.line([(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h),
                       (x, y + h / 2), (x + w / 2, y)], fill=stroke, width=3)
        else:
            draw.rounded_rectangle(rect, radius=14, fill=fill, outline=stroke, width=3)
            if item["dashed"]:
                draw.text((x + 9, y + 6), "待接通", font=small, fill=stroke)
        lines = item["label"].split("\n")
        line_height = 28
        y0 = y + (h - len(lines) * line_height) / 2
        for index, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=body)
            draw.text((x + (w - bbox[2]) / 2, y0 + index * line_height), line,
                      font=body, fill="#1C3040")
    image.save(destination)


def main():
    appointment_flow["name"] = "03 · 当前预约多轮流程"
    pages = [target_architecture, architecture, appointment_flow]
    root = ET.Element("mxfile", {"host": "drawio", "version": "26.0.0",
                                 "type": "device", "compressed": "false"})
    for page in pages:
        root.append(drawio_page(page))
    ET.indent(root, space="  ")
    combined = HERE / "smart-blood-donation.drawio"
    ET.ElementTree(root).write(combined, encoding="utf-8", xml_declaration=True)
    for page in pages:
        preview(page, HERE / f"{page['id']}-preview.png")
    print(f"Wrote {combined} and {len(pages)} PNG previews")


if __name__ == "__main__":
    main()
