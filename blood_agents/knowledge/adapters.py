from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from .rag import Document


class BGEEncoder:
    """bge-large-zh-v1.5 向量化适配器，按需加载模型。"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5") -> None:
        """初始化 BGEEncoder 的依赖与运行状态。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("向量检索需要 sentence-transformers，请安装项目的 rag 可选依赖") from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        """把一批文本编码为向量表示。"""
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_query(self, texts: list[str]):
        """为查询加入 BGE 检索指令后编码；文档编码不加该指令。"""
        # BGE 官方模型卡建议“短问题检索较长文档”时只给 query 加中文指令，
        # 文档本身不加指令；归一化后可用 FAISS 内积近似余弦相似度。
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self.model.encode([instruction + text for text in texts], normalize_embeddings=True)


class BGEReranker:
    """bge-reranker 精排适配器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        """初始化 BGEReranker 的依赖与运行状态。"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("精排需要 sentence-transformers，请安装项目的 rag 可选依赖") from exc

        self.model = CrossEncoder(model_name)

    def __call__(self, query: str, candidates: list[tuple[Document, float]]):
        """用 BGE 交叉编码器重排候选片段，并返回新的相关性分数。"""
        if not candidates:
            return []
        scores = self.model.predict([(query, doc.text) for doc, _ in candidates])
        return sorted(
            [(doc, float(score)) for (doc, _), score in zip(candidates, scores)],
            key=lambda item: item[1],
            reverse=True,
        )


class FAISSVectorStore:
    """本地 FAISS 召回；适合单机开发和小规模知识库。"""

    def __init__(self, documents: list[Document], encoder: BGEEncoder) -> None:
        """初始化 FAISSVectorStore 的依赖与运行状态。"""
        if not documents:
            raise ValueError("FAISS 入库需要至少一个知识片段")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS 检索需要 faiss-cpu，请安装项目的 rag 可选依赖") from exc

        self.documents = documents
        self.encoder = encoder
        vectors = encoder.encode([doc.text for doc in documents]).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 8) -> list[tuple[Document, float]]:
        """检索与用户问题相关的知识片段。"""
        vector = self.encoder.encode_query([query]).astype("float32")
        scores, indices = self.index.search(vector, min(top_k, len(self.documents)))
        return [
            (self.documents[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


class MilvusVectorStore:
    """显式启用的 Milvus 向量库；语料变化时使用新 collection，不清空旧数据。"""

    def __init__(
        self,
        documents: list[Document],
        encoder: BGEEncoder,
        uri: str = "./data/blood_knowledge.db",
        collection_name: str = "blood_knowledge",
        token: Optional[str] = None,
    ) -> None:
        """初始化 MilvusVectorStore 的依赖与运行状态。"""
        if not documents:
            raise ValueError("Milvus 入库需要至少一个知识片段")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,180}", collection_name):
            raise ValueError("MILVUS_COLLECTION 只能包含字母、数字、下划线，并以字母或下划线开头")
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("Milvus 检索需要 pymilvus[milvus-lite]，请安装项目的 rag 可选依赖") from exc

        # collection 名包含模型和全文语料指纹。修改文档/切换模型只会生成新索引，
        # 绝不 drop 已有 collection；持久化文件及历史索引由使用者自行管理。
        payload = [(doc.doc_id, doc.source, doc.text) for doc in documents]
        if len({doc.doc_id for doc in documents}) != len(documents):
            raise ValueError("Milvus 入库的知识片段 ID 不能重复")
        fingerprint = hashlib.sha256(json.dumps(
            [getattr(encoder, "model_name", type(encoder).__name__), payload],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:16]
        self.documents = dict(enumerate(documents))
        self.encoder = encoder
        self.collection_name = f"{collection_name}_{fingerprint}"
        if not uri.startswith(("http://", "https://")):
            local_path = Path(uri).expanduser().resolve()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            uri = str(local_path)
        self.client = MilvusClient(uri=uri, **({"token": token} if token else {}))

        if self.client.has_collection(collection_name=self.collection_name):
            # Milvus Lite 进程重启后 collection 会处于 released 状态；先 load，
            # 否则连检查已有 ID 的 get/query 都会失败。
            self.client.load_collection(collection_name=self.collection_name)
            # 完整索引直接复用；若上次入库中断，只补缺失 ID，避免重复写入。
            existing = {int(row["id"]) for row in self.client.get(
                collection_name=self.collection_name, ids=list(self.documents),
                output_fields=["id"],
            )}
            missing = [index for index in self.documents if index not in existing]
        else:
            missing = list(self.documents)
        if missing:
            vectors = encoder.encode([self.documents[index].text for index in missing])
            if len(vectors.shape) != 2 or vectors.shape[0] != len(missing):
                raise ValueError("BGE 编码器返回的知识片段向量维度异常")
            dimension = int(vectors.shape[1])
            if not self.client.has_collection(collection_name=self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name, dimension=dimension,
                    metric_type="IP", auto_id=False,
                )
            else:
                fields = self.client.describe_collection(collection_name=self.collection_name)["fields"]
                vector_fields = [field for field in fields if field.get("name") == "vector"]
                if len(vector_fields) != 1 or int(vector_fields[0]["params"]["dim"]) != dimension:
                    raise ValueError("已有 Milvus collection 的向量维度与当前编码器不符")
            self.client.insert(collection_name=self.collection_name, data=[
                {"id": index, "vector": vector.astype("float32").tolist()}
                for index, vector in zip(missing, vectors)
            ])

    def search(self, query: str, top_k: int = 8) -> list[tuple[Document, float]]:
        """检索与用户问题相关的知识片段。"""
        if top_k <= 0:
            return []
        vector = self.encoder.encode_query([query])[0].astype("float32").tolist()
        rows = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=min(top_k, len(self.documents)),
            output_fields=["id"],
        )[0]
        return [
            (self.documents[int(row["id"])], float(row["distance"]))
            for row in rows
            if int(row["id"]) in self.documents
        ]
