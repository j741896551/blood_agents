"""教学用 RAG 入库入口：原文 PDF 先暂存审核，获批 Markdown 才建向量索引。

PDF 提取结果不自动加入 corpus.json。原因是原文可能有扫描页、表格、旧版条款，
直接向量化全文会把未核对或与现行规程冲突的片段交给 FAQ 智能体。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    """把用户看到的 PDF 物理页码（从 1 起）转换成 pypdf 的下标。"""
    if page_count < 1:
        raise ValueError("PDF 没有页面")
    if not spec:
        return list(range(page_count))
    selected: set[int] = set()
    for part in spec.split(","):
        bounds = part.strip().split("-")
        if len(bounds) not in (1, 2) or any(not value.isdecimal() for value in bounds):
            raise ValueError("页码格式应为 1,3-5")
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 1 or end < start or end > page_count:
            raise ValueError(f"页码必须在 1 到 {page_count} 之间，且范围递增")
        selected.update(range(start - 1, end))
    return sorted(selected)


def read_pdf(path: Path, page_spec: str | None) -> dict:
    """只读提取；把扫描页/提取为空的情况显式暴露给审核者。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("请先安装 .[rag_ingest]，再提取 PDF") from exc
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("需要现有的本地 PDF 文件")
    reader = PdfReader(str(path))
    pages = parse_pages(page_spec, len(reader.pages))
    extracts = [
        {"page": index + 1, "text": (reader.pages[index].extract_text() or "").replace("\x00", "").strip()}
        for index in pages
    ]
    return {
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total_pages": len(reader.pages),
        "selected_pages": [entry["page"] for entry in extracts],
        "extracted_chars": sum(len(entry["text"]) for entry in extracts),
        "empty_pages": [entry["page"] for entry in extracts if not entry["text"]],
        "extracts": extracts,
    }


def stage_pdf(path: Path, page_spec: str, output: Path, source_url: str) -> dict:
    """生成待审核草稿；使用排他创建，绝不覆盖人工已修订的文件。"""
    if not source_url.startswith("https://"):
        raise ValueError("来源 URL 必须是已核对的 HTTPS 官方地址")
    if output.suffix.lower() != ".md":
        raise ValueError("待审核草稿应为 Markdown 文件")
    report = read_pdf(path, page_spec)
    if not report["extracted_chars"]:
        raise ValueError("选中页面未提取到文字；可能是扫描件，需要 OCR 和人工校对")
    body = [
        "# 待审核的 PDF 文字提取草稿",
        "",
        "<!-- 不能直接加入 corpus.json；先人工核对版本、表格、页码和医学表述。 -->",
        f"- 官方来源：{source_url}",
        f"- 原始 PDF SHA-256：{report['sha256']}",
        f"- PDF 物理页码：{', '.join(map(str, report['selected_pages']))}",
        "- 状态：待人工审核，未进入 RAG",
    ]
    for item in report["extracts"]:
        body.extend(("", f"## PDF 第 {item['page']} 页", "", item["text"] or "【无可提取文字】"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(body) + "\n")
    return {"draft": str(output.resolve()), "pages": report["selected_pages"],
            "chars": report["extracted_chars"], "approved": False}


def index_approved(uri: Path, collection: str, query: str, top_k: int) -> dict:
    """从已批准清单建 Milvus Lite 索引，而非从下载目录或待审核草稿建索引。"""
    if uri.suffix.lower() != ".db":
        raise ValueError("当前入库命令只允许显式指定本地 .db Milvus Lite 文件")
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    from .adapters import BGEEncoder, MilvusVectorStore
    from .rag import HybridRetriever, load_approved_corpus

    documents = load_approved_corpus()
    encoder = BGEEncoder()
    store = MilvusVectorStore(documents, encoder, uri=str(uri), collection_name=collection)
    # get 返回已实际写入/复用的 ID，不能仅以 Python 中的片段数量冒充入库成功。
    indexed = store.client.get(collection_name=store.collection_name,
                               ids=list(store.documents), output_fields=["id"])
    retriever = HybridRetriever(documents, dense_search=store.search, min_score=0.15,
                                dense_min_score=0.55, backend="milvus")
    hits = retriever.search(query, top_k=top_k)
    return {
        "database": str(uri.resolve()), "collection": store.collection_name,
        "approved_documents": len(json.loads(Path(__file__).with_name("corpus.json").read_text(
            encoding="utf-8"))),
        "chunks": len(documents), "indexed_chunks": len(indexed), "model": encoder.model_name,
        "query": query,
        "hits": [{"title": hit["title"], "source": hit["source"],
                  "score": hit["score"], "text_preview": hit["text"][:160]} for hit in hits],
    }


def main() -> None:
    """读取知识文档并建立检索索引。"""
    parser = argparse.ArgumentParser(description="官方 PDF 审核与 Milvus Lite 入库教学工具")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="检查 PDF 页数、提取字数和文本预览")
    inspect.add_argument("pdf", type=Path)
    inspect.add_argument("--pages", help="PDF 物理页码，例如 130-131 或 1,3-5")
    stage = commands.add_parser("stage", help="抽取指定页面为待人工审核草稿，不自动入库")
    stage.add_argument("pdf", type=Path)
    stage.add_argument("--pages", required=True)
    stage.add_argument("--source-url", required=True)
    stage.add_argument("--output", required=True, type=Path)
    index = commands.add_parser("index", help="把 corpus.json 已批准 Markdown 建入 Milvus Lite")
    index.add_argument("--uri", required=True, type=Path)
    index.add_argument("--collection", default="blood_knowledge_demo")
    index.add_argument("--query", default="全血献血后多久可以再献单采血小板")
    index.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            report = read_pdf(args.pdf, args.pages)
            report["preview"] = [
                {"page": item["page"], "text": item["text"][:350]}
                for item in report.pop("extracts")
            ]
        elif args.command == "stage":
            report = stage_pdf(args.pdf, args.pages, args.output, args.source_url)
        else:
            report = index_approved(args.uri, args.collection, args.query, args.top_k)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.exit(2, f"处理失败：{exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
