"""多轮评测的 Excel 输入/输出边界。

只支持本项目公开的三张输入表及普通 .xlsx 单元格，不引入部署时额外依赖。
用户编辑的公式、外部链接和异常大的压缩包一律拒绝；话术只作为纯文本读取。
"""

from __future__ import annotations

import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from blood_agents.log.trace_logging import mask_sensitive

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("", MAIN)
ET.register_namespace("r", REL_DOC)

INPUT_HEADERS = {
    "场景": ["场景ID", "场景说明", "用户ID", "预置四状态预约", "对话评审目标"],
    "对话": ["场景ID", "轮次ID", "用户话术"],
    "断言": ["场景ID", "轮次ID", "检查项", "预期值"],
}
SCALAR_FIELDS = {
    "意图": "intent",
    "动作": "action",
    "结果状态": "result_status",
    "结果码": "result_code",
    "操作成功": "result_success",
    "操作": "operation",
    "数据状态": "data_status",
    "智能体状态": "agent_status",
    "活动智能体": "active_agent",
    "回复模式": "reply_mode",
    "可确认预览": "preview_confirmable",
    "预约号同前轮": "same_appointment_id_as",
}
LIST_FIELDS = {
    "预约状态": "appointment_statuses",
    "工具顺序": "tools_exact",
    "必须调用工具": "tools_include",
    "禁止调用工具": "tools_exclude",
    "回复包含": "answer_contains",
    "回复不包含": "answer_not_contains",
    "节点顺序": "nodes_exact",
    "必须经过节点": "nodes_include",
    "禁止经过节点": "nodes_exclude",
}
SLOT_FIELDS = {"槽位.日期": "date", "槽位.网点": "site", "槽位.时段": "time_slot"}


def _tag(namespace: str, name: str) -> str:
    """生成带命名空间的 XML 标签名。"""
    return f"{{{namespace}}}{name}"


def _plain(value) -> str:
    """提取单元格中可读的纯文本。"""
    return "" if value is None else str(value).strip()


def _as_bool(value, where: str) -> bool:
    """把表格单元格转换为布尔值。"""
    if isinstance(value, bool):
        return value
    normalized = _plain(value).lower()
    if normalized in {"是", "true", "1", "成功"}:
        return True
    if normalized in {"否", "false", "0", "失败"}:
        return False
    raise ValueError(f"{where} 须填写 是/否 或 TRUE/FALSE，实际为 {value!r}")


def _column_index(reference: str) -> int:
    """把 Excel 列名转换为从零开始的列索引。"""
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"无效的单元格地址：{reference}")
    result = 0
    for letter in letters.group():
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _read_xml(archive: zipfile.ZipFile, path: str) -> ET.Element:
    """从 Excel 压缩包读取指定 XML 成员。"""
    return ET.fromstring(archive.read(path))


def _sheet_rows(archive: zipfile.ZipFile, path: str, strings: list[str]) -> list[list]:
    """解析工作表 XML 中的行和单元格。"""
    root = _read_xml(archive, path)
    rows = []
    for row in root.findall(f".//{_tag(MAIN, 'sheetData')}/{_tag(MAIN, 'row')}"):
        if len(rows) >= 5000:
            raise ValueError("单张工作表超过 5000 行，不适合作为对话用例")
        values = []
        for cell in row.findall(_tag(MAIN, "c")):
            if cell.find(_tag(MAIN, "f")) is not None:
                raise ValueError(f"输入表不允许公式：{path}!{cell.get('r')}")
            index = _column_index(cell.get("r", ""))
            if index > 100:
                raise ValueError("输入表列数超过 100 列")
            values.extend([None] * max(0, index + 1 - len(values)))
            kind = cell.get("t")
            if kind == "inlineStr":
                value = "".join(part.text or "" for part in cell.findall(f".//{_tag(MAIN, 't')}"))
            else:
                raw = cell.findtext(_tag(MAIN, "v"))
                if kind == "s":
                    value = strings[int(raw)] if raw is not None else ""
                elif kind == "b":
                    value = raw == "1"
                elif raw is None:
                    value = None
                elif kind in {"str", "e"}:
                    value = raw
                else:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
            values[index] = value
        rows.append(values)
    return rows


def read_input_tables(path: str | Path) -> dict[str, list[list]]:
    """仅解析有界的工作表数据；拒绝公式/外链，保留 Excel 行顺序。"""
    source = Path(path)
    if source.suffix.lower() != ".xlsx":
        raise ValueError("Excel 用例文件须为 .xlsx")
    if source.stat().st_size > 5_000_000:
        raise ValueError("Excel 用例文件超过 5 MB")
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if sum(item.file_size for item in entries) > 20_000_000:
            raise ValueError("Excel 解压后超过 20 MB")
        if any(item.flag_bits & 1 for item in entries):
            raise ValueError("不支持加密 Excel")
        if any(
            "externalLink" in item.filename or item.filename.endswith("vbaProject.bin")
            for item in entries
        ):
            raise ValueError("输入表不能包含外部链接或宏")
        workbook = _read_xml(archive, "xl/workbook.xml")
        relationships = _read_xml(archive, "xl/_rels/workbook.xml.rels")
        targets = {item.get("Id"): item.get("Target") for item in relationships}
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = _read_xml(archive, "xl/sharedStrings.xml")
            strings = [
                "".join(t.text or "" for t in item.findall(f".//{_tag(MAIN, 't')}"))
                for item in shared.findall(_tag(MAIN, "si"))
            ]
        tables = {}
        for item in workbook.findall(f".//{_tag(MAIN, 'sheets')}/{_tag(MAIN, 'sheet')}"):
            name = item.get("name", "")
            if name not in INPUT_HEADERS:
                continue
            target = targets.get(item.get(_tag(REL_DOC, "id")))
            if not target:
                raise ValueError(f"工作表 {name} 缺少文件关联")
            member = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join("xl", target))
            )
            if not member.startswith("xl/worksheets/"):
                raise ValueError(f"工作表 {name} 的文件路径不合法")
            tables[name] = _sheet_rows(archive, member, strings)
    for name, headers in INPUT_HEADERS.items():
        if name not in tables:
            raise ValueError(f"Excel 缺少“{name}”工作表")
        actual = (
            [_plain(value) for value in tables[name][0][: len(headers)]] if tables[name] else []
        )
        if actual != headers and not (name == "场景" and actual == headers[:4]):
            raise ValueError(f"“{name}”首行列名须为：{'、'.join(headers)}")
    return tables


def suite_data_from_tables(tables: dict[str, list[list]], name: str) -> dict:
    """三张人类可编辑表转为已有 Pydantic 用例契约，复用原验收逻辑。"""
    cases = []
    by_case = {}
    for row_number, row in enumerate(tables["场景"][1:], 2):
        if not any(_plain(value) for value in row):
            continue
        case_id = _plain(row[0] if len(row) > 0 else None)
        if not case_id or case_id in by_case:
            raise ValueError(f"场景第 {row_number} 行的场景ID为空或重复")
        case = {
            "id": case_id,
            "description": _plain(row[1] if len(row) > 1 else None),
            "review_goal": _plain(row[4] if len(row) > 4 else None),
            "user_id": _plain(row[2] if len(row) > 2 else None),
            "seed_appointment_samples": _as_bool(row[3], f"场景第 {row_number} 行预置四状态预约")
            if len(row) > 3 and _plain(row[3])
            else False,
            "turns": [],
        }
        by_case[case_id] = case
        cases.append(case)
    if not cases:
        raise ValueError("Excel 场景表没有用例")
    by_turn = {}
    for row_number, row in enumerate(tables["对话"][1:], 2):
        if not any(_plain(value) for value in row):
            continue
        case_id = _plain(row[0] if len(row) > 0 else None)
        turn_id = _plain(row[1] if len(row) > 1 else None)
        if case_id not in by_case or not turn_id or (case_id, turn_id) in by_turn:
            raise ValueError(f"对话第 {row_number} 行的场景/轮次ID不存在、为空或重复")
        message = row[2] if len(row) > 2 else None
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"对话第 {row_number} 行缺少用户话术")
        turn = {"id": turn_id, "message": message}
        by_case[case_id]["turns"].append(turn)
        by_turn[(case_id, turn_id)] = turn
    for row_number, row in enumerate(tables["断言"][1:], 2):
        if not any(_plain(value) for value in row):
            continue
        key = (_plain(row[0] if len(row) > 0 else None), _plain(row[1] if len(row) > 1 else None))
        if key not in by_turn:
            raise ValueError(f"断言第 {row_number} 行未找到对应对话轮次：{key}")
        label = _plain(row[2] if len(row) > 2 else None)
        value = row[3] if len(row) > 3 else None
        if not label or value is None or value == "":
            raise ValueError(f"断言第 {row_number} 行检查项或预期值为空")
        expected = by_turn[key].setdefault("expected", {})
        if label in SCALAR_FIELDS:
            field = SCALAR_FIELDS[label]
            if field in expected:
                raise ValueError(f"断言第 {row_number} 行重复设置 {label}")
            expected[field] = (
                _as_bool(value, f"断言第 {row_number} 行")
                if field in {"result_success", "preview_confirmable"}
                else _plain(value)
            )
        elif label in LIST_FIELDS:
            expected.setdefault(LIST_FIELDS[label], []).append(_plain(value))
        elif label in SLOT_FIELDS:
            slots = expected.setdefault("slots", {})
            field = SLOT_FIELDS[label]
            if field in slots:
                raise ValueError(f"断言第 {row_number} 行重复设置 {label}")
            slots[field] = _plain(value)
        else:
            raise ValueError(f"断言第 {row_number} 行未知检查项：{label}")
    for case in cases:
        if not case["turns"]:
            raise ValueError(f"场景 {case['id']} 没有对话轮次")
    return {"version": 1, "name": name, "cases": cases}


def _column_letter(index: int) -> str:
    """把列序号转换为 Excel 列字母。"""
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _xml_bytes(element: ET.Element) -> bytes:
    """把 XML 元素序列化为带声明的 Excel 工作簿成员字节。"""
    # OOXML 的 package relationship/content-type 根命名空间应以默认命名空间写出；
    # 某些严格的表格读取器虽能解析 ns0 前缀，但无法据此识别包部件。
    if element.tag.startswith("{"):
        ET.register_namespace("", element.tag[1:].split("}", 1)[0])
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _sheet_xml(
    rows: list[list],
    widths: list[int],
    *,
    filter_row: int | None = None,
    freeze_rows: int = 0,
    status_col: int | None = None,
) -> bytes:
    """生成工作表内容对应的 XML。"""
    root = ET.Element(_tag(MAIN, "worksheet"))
    views = ET.SubElement(root, _tag(MAIN, "sheetViews"))
    view = ET.SubElement(views, _tag(MAIN, "sheetView"), workbookViewId="0", showGridLines="0")
    if freeze_rows:
        ET.SubElement(
            view,
            _tag(MAIN, "pane"),
            ySplit=str(freeze_rows),
            topLeftCell=f"A{freeze_rows + 1}",
            state="frozen",
        )
    columns = ET.SubElement(root, _tag(MAIN, "cols"))
    for index, width in enumerate(widths, 1):
        ET.SubElement(
            columns,
            _tag(MAIN, "col"),
            min=str(index),
            max=str(index),
            width=str(width),
            customWidth="1",
        )
    data = ET.SubElement(root, _tag(MAIN, "sheetData"))
    for row_number, values in enumerate(rows, 1):
        row = ET.SubElement(data, _tag(MAIN, "row"), r=str(row_number))
        if row_number > 1:
            row.set("ht", "40" if status_col else "28")
            row.set("customHeight", "1")
        for index, value in enumerate(values, 1):
            if value is None or value == "":
                continue
            style = (
                1
                if row_number == 1
                or (status_col is None and len(rows[0]) == 2 and values[0] == "场景ID")
                else 0
            )
            if status_col == index and row_number > 1:
                style = {
                    "通过": 2,
                    "失败": 3,
                    "待评": 4,
                    "跳过": 4,
                    "不可评审": 4,
                    "未启用": 4,
                }.get(str(value), 0)
            cell = ET.SubElement(
                row, _tag(MAIN, "c"), r=f"{_column_letter(index)}{row_number}", s=str(style)
            )
            if isinstance(value, bool):
                cell.set("t", "b")
                ET.SubElement(cell, _tag(MAIN, "v")).text = "1" if value else "0"
            elif isinstance(value, (int, float)):
                ET.SubElement(cell, _tag(MAIN, "v")).text = str(value)
            else:
                cell.set("t", "inlineStr")
                inline = ET.SubElement(cell, _tag(MAIN, "is"))
                content = ET.SubElement(inline, _tag(MAIN, "t"))
                content.text = str(value)
                if str(value).strip() != str(value):
                    content.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    if filter_row:
        ET.SubElement(
            root,
            _tag(MAIN, "autoFilter"),
            ref=f"A{filter_row}:{_column_letter(max(len(row) for row in rows))}{len(rows)}",
        )
    return _xml_bytes(root)


def _styles_xml() -> bytes:
    """生成 Excel 工作簿的样式 XML。"""
    root = ET.Element(_tag(MAIN, "styleSheet"))
    fonts = ET.SubElement(root, _tag(MAIN, "fonts"), count="2")
    for bold in (False, True):
        font = ET.SubElement(fonts, _tag(MAIN, "font"))
        ET.SubElement(font, _tag(MAIN, "sz"), val="10")
        if bold:
            ET.SubElement(font, _tag(MAIN, "b"))
            ET.SubElement(font, _tag(MAIN, "color"), rgb="FFFFFFFF")
        ET.SubElement(font, _tag(MAIN, "name"), val="Arial")
    fills = ET.SubElement(root, _tag(MAIN, "fills"), count="6")
    ET.SubElement(
        ET.SubElement(fills, _tag(MAIN, "fill")), _tag(MAIN, "patternFill"), patternType="none"
    )
    ET.SubElement(
        ET.SubElement(fills, _tag(MAIN, "fill")), _tag(MAIN, "patternFill"), patternType="gray125"
    )
    for color in ("FF203B5A", "FFE5F4EA", "FFFBE5E5", "FFFFF2D5"):
        pattern = ET.SubElement(
            ET.SubElement(fills, _tag(MAIN, "fill")), _tag(MAIN, "patternFill"), patternType="solid"
        )
        ET.SubElement(pattern, _tag(MAIN, "fgColor"), rgb=color)
        ET.SubElement(pattern, _tag(MAIN, "bgColor"), indexed="64")
    borders = ET.SubElement(root, _tag(MAIN, "borders"), count="1")
    ET.SubElement(borders, _tag(MAIN, "border"))
    cell_style = ET.SubElement(root, _tag(MAIN, "cellStyleXfs"), count="1")
    ET.SubElement(cell_style, _tag(MAIN, "xf"), numFmtId="0", fontId="0", fillId="0", borderId="0")
    formats = ET.SubElement(root, _tag(MAIN, "cellXfs"), count="5")
    for font_id, fill_id in ((0, 0), (1, 2), (0, 3), (0, 4), (0, 5)):
        xf = ET.SubElement(
            formats,
            _tag(MAIN, "xf"),
            numFmtId="0",
            fontId=str(font_id),
            fillId=str(fill_id),
            borderId="0",
            xfId="0",
            applyAlignment="1",
        )
        ET.SubElement(xf, _tag(MAIN, "alignment"), vertical="center", wrapText="1")
    styles = ET.SubElement(root, _tag(MAIN, "cellStyles"), count="1")
    ET.SubElement(styles, _tag(MAIN, "cellStyle"), name="Normal", xfId="0", builtinId="0")
    return _xml_bytes(root)


def write_workbook(
    path: str | Path, sheets: list[tuple[str, list[list], list[int], int | None]]
) -> Path:
    """导出静态报告；全部文本使用 inlineStr，`=...` 不会被当作 Excel 公式。"""
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    workbook = ET.Element(_tag(MAIN, "workbook"))
    book_sheets = ET.SubElement(workbook, _tag(MAIN, "sheets"))
    relations = ET.Element(_tag(REL_PKG, "Relationships"))
    types = ET.Element(_tag(CONTENT, "Types"))
    ET.SubElement(
        types,
        _tag(CONTENT, "Default"),
        Extension="rels",
        ContentType="application/vnd.openxmlformats-package.relationships+xml",
    )
    ET.SubElement(types, _tag(CONTENT, "Default"), Extension="xml", ContentType="application/xml")
    ET.SubElement(
        types,
        _tag(CONTENT, "Override"),
        PartName="/xl/workbook.xml",
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
    )
    ET.SubElement(
        types,
        _tag(CONTENT, "Override"),
        PartName="/xl/styles.xml",
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml",
    )
    with (
        target.open("xb") as raw,
        zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        root_rels = ET.Element(_tag(REL_PKG, "Relationships"))
        ET.SubElement(
            root_rels,
            _tag(REL_PKG, "Relationship"),
            Id="rId1",
            Type=f"{REL_DOC}/officeDocument",
            Target="xl/workbook.xml",
        )
        archive.writestr("_rels/.rels", _xml_bytes(root_rels))
        for index, (name, rows, widths, status_col) in enumerate(sheets, 1):
            ET.SubElement(
                book_sheets,
                _tag(MAIN, "sheet"),
                name=name,
                sheetId=str(index),
                attrib={_tag(REL_DOC, "id"): f"rId{index}"},
            )
            ET.SubElement(
                relations,
                _tag(REL_PKG, "Relationship"),
                Id=f"rId{index}",
                Type=f"{REL_DOC}/worksheet",
                Target=f"worksheets/sheet{index}.xml",
            )
            ET.SubElement(
                types,
                _tag(CONTENT, "Override"),
                PartName=f"/xl/worksheets/sheet{index}.xml",
                ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
            )
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _sheet_xml(
                    rows,
                    widths,
                    filter_row=1 if len(rows) > 1 and name != "总览" else None,
                    freeze_rows=1,
                    status_col=status_col,
                ),
            )
        ET.SubElement(
            relations,
            _tag(REL_PKG, "Relationship"),
            Id=f"rId{len(sheets) + 1}",
            Type=f"{REL_DOC}/styles",
            Target="styles.xml",
        )
        archive.writestr("xl/workbook.xml", _xml_bytes(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", _xml_bytes(relations))
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr("[Content_Types].xml", _xml_bytes(types))
    target.chmod(0o600)
    return target


def write_report_xlsx(report: dict, path: str | Path) -> Path:
    """总览、逐轮对话和执行证据分开呈现，保留 JSON 供程序追溯。"""
    clean = mask_sensitive(report)
    summary = clean["summary"]
    mode = "真实模型" if clean["mode"] == "model" else "离线规则 + Mock"
    scope = (
        "检验模型理解、回复和 Mock 业务断言"
        if clean["mode"] == "model"
        else "仅检验规则/Mock 编排，不检验对话质量"
    )
    overview = [
        ["多轮对话评测结果", "值"],
        ["测试集", clean["suite"]],
        ["运行ID", clean["run_id"]],
        ["用例内容指纹（SHA-256）", clean.get("suite_sha256") or "未记录"],
        ["运行时间", clean["started_at"]],
        ["运行模式", mode],
        ["判定范围", scope],
        ["场景总数", summary["cases_total"]],
        ["场景通过", summary["cases_passed"]],
        ["轮次总数", summary["turns_total"]],
        ["轮次通过", summary["turns_passed"]],
        ["待评轮次", summary["turns_ungraded"]],
        ["仅模型路径已检查轮次", summary.get("turns_model_only", 0)],
        ["LLM 回复轮次", summary["llm_replies"]],
        ["结构与模型调用门禁通过", "是" if summary["structural_passed"] else "否"],
        ["对话评审", "已启用" if clean.get("review_mode") == "model" else "未启用"],
        ["评审通过场景", summary.get("review_cases_passed", 0)],
        [
            "对话评审全通过",
            "未评审"
            if summary.get("review_passed") is None
            else ("是" if summary["review_passed"] else "否"),
        ],
        [
            "综合通过",
            "未评审"
            if summary.get("overall_passed") is None
            else ("是" if summary["overall_passed"] else "否"),
        ],
        ["说明", "综合通过须同时满足逐轮模型调用门禁、显式断言和整场模型评审。"],
        ["场景ID", "场景说明", "用户ID", "结构断言", "对话评审", "轮次", "失败轮次"],
    ]
    dialogue = [
        [
            "场景ID",
            "轮次ID",
            "结构判定",
            "用户话术",
            "智能体回复",
            "预期断言",
            "实际意图",
            "业务结果",
            "失败或待评原因",
            "调用工具",
            "回复模式",
            "耗时(ms)",
        ]
    ]
    evidence = [
        [
            "场景ID",
            "轮次ID",
            "追踪ID",
            "会话ID",
            "导航模式",
            "预约理解模式",
            "回复模式",
            "节点路径",
            "工具轨迹",
            "业务结果详情",
        ]
    ]
    reviews = [
        [
            "场景ID",
            "评审目标",
            "对话评审",
            "任务完成",
            "总评原因",
            "问题轮次",
            "问题类别",
            "原文证据",
            "问题分析",
            "模型服务",
        ]
    ]
    labels = {
        "passed": "通过", "model_checked": "模型路径通过，业务未断言",
        "failed": "失败", "ungraded": "待评", "skipped": "跳过",
    }
    review_labels = {
        "passed": "通过",
        "failed": "失败",
        "uncertain": "待评",
        "unavailable": "不可评审",
        "not_requested": "未启用",
    }
    for case in clean["cases"]:
        failures = sum(turn["verdict"] in {"failed", "skipped"} for turn in case["turns"])
        review = case.get("review") or {}
        review_status = review_labels.get(review.get("status"), "待评")
        overview.append(
            [
                case["case_id"],
                case["description"],
                case["user_id"],
                "通过" if case["passed"] else "未通过",
                review_status,
                len(case["turns"]),
                failures,
            ]
        )
        for issue in review.get("issues") or [None]:
            reviews.append(
                [
                    case["case_id"],
                    case.get("review_goal", ""),
                    review_status,
                    "是"
                    if review.get("task_completed")
                    else "否"
                    if review.get("task_completed") is False
                    else "未评审",
                    review.get("reason", ""),
                    (issue or {}).get("turn_id", ""),
                    (issue or {}).get("category", ""),
                    (issue or {}).get("evidence_quote", ""),
                    (issue or {}).get("reason", ""),
                    review.get("provider") or "",
                ]
            )
        for turn in case["turns"]:
            result = turn.get("result") or {}
            model = turn.get("model_info") or {}
            response = turn.get("response_info") or {}
            tools = turn.get("tool_trace") or []
            failure = "；".join(turn.get("failures") or [])
            if turn["verdict"] == "ungraded":
                failure = "未填写预期断言，需人工复核"
            dialogue.append(
                [
                    case["case_id"],
                    turn["turn_id"],
                    labels[turn["verdict"]],
                    turn["message"],
                    turn.get("answer", ""),
                    json.dumps(turn.get("expected"), ensure_ascii=False)
                    if turn.get("expected")
                    else "未填写",
                    turn.get("intent", ""),
                    result.get("status") or result.get("code") or "",
                    failure,
                    "、".join(item.get("tool", "") for item in tools),
                    response.get("mode", ""),
                    turn.get("elapsed_ms"),
                ]
            )
            evidence.append(
                [
                    case["case_id"],
                    turn["turn_id"],
                    turn.get("trace_id", ""),
                    case["session_id"],
                    (model.get("navigation") or {}).get("mode", ""),
                    (model.get("appointment") or {}).get("mode", ""),
                    response.get("mode", ""),
                    " → ".join(turn.get("node_path") or []),
                    json.dumps(tools, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ]
            )
    return write_workbook(
        path,
        [
            ("总览", overview, [26, 88, 26, 18, 18, 14, 16], None),
            ("对话结果", dialogue, [25, 25, 12, 52, 90, 65, 18, 25, 65, 45, 16, 15], 3),
            ("对话评审", reviews, [25, 70, 14, 15, 90, 25, 25, 70, 80, 20], 3),
            ("执行证据", evidence, [25, 25, 38, 48, 18, 18, 16, 90, 90, 90], None),
        ],
    )
