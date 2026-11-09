"""献血知识检索：从有来源的本地摘要加载，当前默认仍为轻量关键词召回。"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str
    metadata: dict


class MultiStrategyChunker:
    """按标题/段落/滑窗组合切分，保留来源元数据。"""

    def split(self, text: str, source: str, size: int = 180, overlap: int = 30,
              metadata: dict | None = None, doc_id_prefix: str | None = None) -> list[Document]:
        """按配置把知识文档切成可检索片段。"""
        sections = [part.strip() for part in re.split(r"\n(?=#{1,6}\s|第[一二三四五六七八九十]+[章节])", text) if part.strip()]
        chunks: list[Document] = []
        for section_index, section in enumerate(sections or [text]):
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            for paragraph_index, paragraph in enumerate(paragraphs or [section]):
                start = 0
                while start < len(paragraph):
                    chunk = paragraph[start : start + size]
                    # 不用来源 URL 单独作为主键：同一官方页面可能对应清单中的
                    # 多个独立主题，URL 相同时片段 ID 仍必须唯一。
                    doc_id = f"{doc_id_prefix or source}:{section_index}:{paragraph_index}:{start}"
                    chunks.append(Document(doc_id, chunk, source, {
                        **(metadata or {}), "offset": start,
                    }))
                    if start + size >= len(paragraph):
                        break
                    start += max(1, size - overlap)
        return chunks


class HybridRetriever:
    """轻量离线实现；生产可将 dense_search 接到 FAISS/Milvus + bge-large-zh。"""

    def __init__(self, documents: list[Document], dense_search=None, reranker=None,
                 min_score: float = 0.0, dense_min_score: float = 0.0,
                 backend: str | None = None) -> None:
        """初始化 HybridRetriever 的依赖与运行状态。"""
        self.documents = documents
        self.dense_search = dense_search
        self.reranker = reranker
        self.min_score = min_score
        self.dense_min_score = dense_min_score
        self.backend = backend or ("hybrid" if dense_search else "memory")

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """计算问题与知识片段的词面匹配分数。"""
        query_terms = set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", query.lower()))
        text_terms = set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower()))
        return len(query_terms & text_terms) / math.sqrt(max(1, len(query_terms) * len(text_terms)))

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """检索与用户问题相关的知识片段。"""
        if top_k <= 0:
            return []
        # 第一阶段：关键词召回，保证没有向量模型时也可以离线运行。
        candidates = [(doc, self._lexical_score(query, doc.text)) for doc in self.documents]
        if self.dense_search:
            # 第二阶段：可选的 bge + FAISS/Milvus 向量召回并融合分数。
            dense = self.dense_search(query, top_k * 2)
            # 内积分数必须先过绝对阈值；否则无关问题在向量库中总有最近邻，
            # 若直接加分会把“最不坏的候选”误当成知识证据。
            scores = {doc.doc_id: score for doc, score in dense
                      if math.isfinite(score) and score >= self.dense_min_score}
            candidates = [(doc, score + scores.get(doc.doc_id, 0.0)) for doc, score in candidates]
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[: top_k * 2]
        # 先按原始召回强度判断有没有证据。CrossEncoder 的原始分数范围不固定，
        # 只能用于候选之间排序，不能替代本门槛或把弱召回提升成“可信命中”。
        candidates = [(doc, score) for doc, score in candidates
                      if math.isfinite(score) and score >= self.min_score and score > 0]
        rerank_scores: dict[str, float] = {}
        if self.reranker and candidates:
            # 精排器必须只重排已通过门槛的候选；原始分数与来源由检索层保管。
            reranked = self.reranker(query, candidates)
            originals = {doc.doc_id: (doc, score) for doc, score in candidates}
            if len(reranked) != len(originals):
                raise ValueError("精排结果数量与检索候选不一致")
            ordered = []
            for doc, rerank_score in reranked:
                if (doc.doc_id not in originals or doc.doc_id in rerank_scores
                        or not math.isfinite(float(rerank_score))):
                    raise ValueError("精排结果含未知、重复或非法分数的片段")
                rerank_scores[doc.doc_id] = float(rerank_score)
                ordered.append(originals[doc.doc_id])
            candidates = ordered
        return [
            {"doc_id": doc.doc_id, "text": doc.text, "source": doc.source,
             "title": doc.metadata.get("title"), "published_at": doc.metadata.get("published_at"),
             "source_kind": doc.metadata.get("source_kind", "public"),
             "provenance": doc.metadata.get("provenance"),
             "retrieval_backend": self.backend,
             "score": round(score, 4),
             **({"rerank_score": round(rerank_scores[doc.doc_id], 4)}
                if doc.doc_id in rerank_scores else {})}
            for doc, score in candidates[:top_k]
        ]


def load_approved_corpus(path: Path | None = None) -> list[Document]:
    """按清单加载随包 Markdown/旧式内联摘要；不在请求时联网抓取。"""
    corpus_path = path or Path(__file__).with_name("corpus.json")
    records = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("知识库清单必须是文档列表")
    chunks: list[Document] = []
    seen: set[str] = set()
    chunker = MultiStrategyChunker()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("知识库条目格式异常")
        fields = ("id", "title", "source", "published_at", "checked_at", "provenance")
        if any(not isinstance(record.get(field), str) or not record[field].strip()
               for field in fields):
            raise ValueError("知识库条目缺少来源或时间")
        if record["id"] in seen:
            raise ValueError("知识库文档 ID 重复")
        seen.add(record["id"])
        source_kind = record.get("source_kind", "public")
        parsed = urlparse(record["source"])
        if source_kind == "public":
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("公开知识库来源必须是 HTTPS 页面")
        elif source_kind == "synthetic":
            if parsed.scheme != "demo" or parsed.netloc != "blood-station":
                raise ValueError("模拟知识库来源必须使用 demo://blood-station/")
        else:
            raise ValueError("知识库来源类型未知")
        if ("path" in record) == ("text" in record):
            raise ValueError("知识库条目必须且只能提供 path 或 text 正文")
        if "path" in record:
            relative = record["path"]
            if not isinstance(relative, str) or not relative.endswith(".md"):
                raise ValueError("知识库文档路径必须是相对 Markdown 文件")
            # 清单路径虽随包发布，仍限制在清单所在目录内，避免误读其他本地文件。
            root = corpus_path.parent.resolve()
            document_path = (root / relative).resolve()
            if not document_path.is_relative_to(root):
                raise ValueError("知识库文档路径不能越出语料目录")
            body = document_path.read_text(encoding="utf-8")
        else:
            body = record["text"]
        if not isinstance(body, str) or not body.strip():
            raise ValueError("知识库正文不能为空")
        metadata = {field: record[field] for field in fields}
        metadata["source_kind"] = source_kind
        documents = chunker.split(body, record["source"], metadata=metadata,
                                  doc_id_prefix=record["id"])
        if source_kind == "synthetic":
            # 分块可能把文档首页声明切走；每个送往模型的片段都单独标明虚构性质。
            label = "【虚构演示资料，不代表真实血站或公司制度】"
            documents = [Document(doc.doc_id, f"{label}\n{doc.text}", doc.source, doc.metadata)
                         for doc in documents]
        chunks.extend(documents)
    return chunks


def default_retriever() -> HybridRetriever:
    """按环境配置加载语料、向量后端和可选重排器。"""
    # 字符重合极易给无关问题打出微弱正分；当前阈值只是基础噪声过滤，
    # 后续任务 5 仍需用多来源正反例评估召回/误召回，不能视作医学可靠性保证。
    documents = load_approved_corpus()
    backend = os.getenv("VECTOR_BACKEND", "memory").strip().lower()
    rerank_backend = os.getenv("RAG_RERANKER", "none").strip().lower()
    if rerank_backend not in {"none", "bge"}:
        raise ValueError(f"不支持的 RAG_RERANKER={rerank_backend!r}；当前可用 none/bge")

    def configured_reranker():
        """按配置创建可选的检索重排器。"""
        if rerank_backend == "none":
            return None
        from .adapters import BGEReranker

        # 显式开启时才加载约 GB 级模型；默认关键词模式仍保持零模型下载。
        return BGEReranker(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    if backend == "memory":
        return HybridRetriever(documents, reranker=configured_reranker(), min_score=0.15)
    if backend not in {"faiss", "milvus"}:
        raise ValueError(f"不支持的 VECTOR_BACKEND={backend!r}；当前可用 memory/faiss/milvus")
    from .adapters import BGEEncoder, FAISSVectorStore, MilvusVectorStore

    # 仅显式启用向量后端时才加载模型/索引；无配置时不会下载模型。
    threshold = float(os.getenv("RAG_DENSE_MIN_SCORE", "0.55"))
    if not 0 <= threshold <= 1:
        raise ValueError("RAG_DENSE_MIN_SCORE 必须在 0 到 1 之间")
    encoder = BGEEncoder(os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"))
    if backend == "faiss":
        store = FAISSVectorStore(documents, encoder)
    else:
        # 仅显式切到 Milvus 时才连接向量库；本地 Lite URI 持久化到 .db 文件。
        store = MilvusVectorStore(
            documents, encoder,
            # pymilvus 自身也读取 MILVUS_URI，且只接受 HTTP 地址；项目变量另起名，
            # 否则 Milvus Lite 的本地 .db 路径会在 import pymilvus 时被误判非法。
            uri=os.getenv("RAG_MILVUS_URI", "./data/blood_knowledge.db"),
            collection_name=os.getenv("MILVUS_COLLECTION", "blood_knowledge"),
            token=os.getenv("MILVUS_TOKEN") or None,
        )
    return HybridRetriever(documents, dense_search=store.search,
                           reranker=configured_reranker(),
                           min_score=0.15, dense_min_score=threshold, backend=backend)
