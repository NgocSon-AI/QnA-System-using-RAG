import hashlib
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.utils.logger import Logger


class QdrantIngestor:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
        device: str = "cpu",
        log_name: str = "QdrantIngestor",
        reset_collection: bool = False,  # <-- thêm param
    ) -> None:
        self.logger = Logger(name=log_name).get_logger()
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.device = device

        self.logger.info(f"🔧 Embedding model loaded on device: {self.device}")
        # Truyền flag tiếp vào helper
        self._ensure_collection_exists(reset_collection=reset_collection)

    def _ensure_collection_exists(self, reset_collection: bool = False) -> None:
        """Tạo collection nếu chưa tồn tại; nếu reset_collection=True thì xóa + tạo mới."""
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            if reset_collection:
                self.logger.info(f"♻️ Xóa collection `{self.collection_name}` cũ...")
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                self.logger.info(f"✅ Collection `{self.collection_name}` đã tồn tại.")
                return

        self.logger.info(f"🚀 Tạo collection `{self.collection_name}`...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, distance=Distance.COSINE
            ),
        )
        self.logger.info(f"✅ Collection `{self.collection_name}` được tạo thành công.")

    # =========================================================
    # Utility helpers
    # =========================================================
    def _generate_chunk_id(self, source: str, chunk_idx: int, chunk_text: Any) -> str:
        """Sinh ID duy nhất dựa vào nội dung và vị trí chunk"""
        if isinstance(chunk_text, dict):
            chunk_text = chunk_text.get("text", "")
        if not isinstance(chunk_text, str):
            chunk_text = str(chunk_text)

        unique_str = f"{source}-{chunk_idx}-{chunk_text[:100]}"
        return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    # =========================================================
    # Main function
    # =========================================================
    def upsert_to_qdrant(
        self,
        pdf_path: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("❌ Số lượng chunks và embeddings không khớp")

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = self._generate_chunk_id(pdf_path, idx, chunk)
            payload = {
                "text": chunk.get("text", ""),
                "chunk_index": idx,
                "source": chunk.get("source", str(pdf_path)),
                "source_type": "pdf",
                "source_id": Path(pdf_path).stem,
                "page_number": chunk.get("page", None),
                "language": "vi",
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.logger.info(
            f"🚀 Upserting {len(points)} vectors vào `{self.collection_name}`..."
        )
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.logger.info("✅ Upsert hoàn tất.")

    # =========================================================
    # Optional: Kiểm tra số lượng vector hiện tại
    # =========================================================
    def collection_size(self) -> int:
        """Trả về số lượng vector hiện có trong collection"""
        info = self.client.count(collection_name=self.collection_name)
        return info.count
