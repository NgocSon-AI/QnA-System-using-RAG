from qdrant_client import QdrantClient, models
from typing import List, Optional, Dict, Any
from loguru import logger

from config import get_settings
from script_test import get_embedding_model
from preprocess_fn_basic_ngocson import preprocess_vi_for_jina

settings = get_settings()
embedding_model = get_embedding_model()


class QdrantSearcher:
    """
    Lớp QdrantSearcher cung cấp các phương thức tìm kiếm khác nhau
    dựa trên cơ sở dữ liệu vector Qdrant.

    Các kiểu tìm kiếm được hỗ trợ:
        - **Semantic Search**: tìm kiếm theo ngữ nghĩa (dựa trên vector embedding).
        - **Scroll**: phân trang (lấy dữ liệu theo từng phần).
        - **Hybrid Search**: kết hợp giữa tìm kiếm ngữ nghĩa và tìm kiếm từ khóa.

    Attributes:
        client (QdrantClient): Đối tượng client kết nối tới Qdrant.
        collection_name (str): Tên của collection được sử dụng để truy vấn.
    """

    def __init__(
        self,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        collection_name: str = settings.COLLECTION_NAME,
    ):
        """
        Khởi tạo đối tượng QdrantSearcher.

        Args:
            host (str): Địa chỉ host của Qdrant server.
            port (int): Cổng (port) của Qdrant.
            collection_name (str): Tên collection sẽ được truy vấn.
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    # ==================================================
    # ================== SEMANTIC SEARCH ================
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        use_exact: bool = False,
        hnsw_ef_construct: int = 64,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """
        Thực hiện tìm kiếm ngữ nghĩa (semantic search) trong Qdrant.

        Args:
            query (str): Câu truy vấn của người dùng.
            top_k (int, optional): Số lượng kết quả trả về. Mặc định = 5.
            use_exact (bool, optional): Nếu True, tìm kiếm chính xác (không dùng HNSW). Mặc định = False.
            hnsw_ef_construct (int, optional): Tham số tối ưu HNSW. Mặc định = 64.
            filter_payload (dict, optional): Điều kiện lọc payload (VD: {"source": "report.pdf"}).

        Returns:
            List[dict]: Danh sách các điểm (points) phù hợp nhất với truy vấn.
        """
        try:
            # 1. Tiền xử lý truy vấn tiếng Việt
            query_cleaned = preprocess_vi_for_jina(query)

            # 2. Tạo vector embedding cho truy vấn
            query_vector = embedding_model.embed_query(query_cleaned)

            # 3. Thiết lập tham số tìm kiếm
            search_params = models.SearchParams(
                exact=use_exact, hnsw_ef=hnsw_ef_construct if not use_exact else None
            )

            # 4. Thiết lập bộ lọc payload nếu có
            q_filter = None
            if filter_payload:
                must = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
                q_filter = models.Filter(must=must)

            # 5. Thực hiện truy vấn
            logger.info(
                f"Đang tìm kiếm TOP-{top_k} vector trong collection {self.collection_name}."
            )
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=q_filter,
                search_params=search_params,
                limit=top_k,
            )

            # 6. Xử lý và định dạng kết quả trả về
            outputs = []
            for point in results.points:
                payload = point.payload or {}
                outputs.append(
                    {
                        "text": payload.get("text", ""),
                        "score": point.score,
                        "chunk_index": payload.get("chunk_index"),
                        "source": payload.get("source"),
                    }
                )

            return outputs
        except Exception as e:
            logger.error(f"LỖI TRONG QUÁ TRÌNH SEMANTIC SEARCH: {e}")
            return []

    # ==================================================
    # ================== SCROLL ========================
    def scroll(self, limit: int = 10, offset: Optional[int] = None) -> List[dict]:
        """
        Lấy dữ liệu theo dạng phân trang (scroll).

        Args:
            limit (int, optional): Số lượng phần tử mỗi lần lấy. Mặc định = 10.
            offset (int, optional): Vị trí bắt đầu lấy dữ liệu.

        Returns:
            List[dict]: Danh sách các điểm (points) được lấy ra từ collection.
        """
        try:
            results, next_page = self.client.scroll(
                collection_name=self.collection_name, limit=limit, offset=offset
            )

            outputs = []
            for point in results:
                payload = point.payload or {}
                outputs.append(
                    {
                        "id": point.id,
                        "text": payload.get("text", ""),
                        "source": payload.get("source"),
                    }
                )
            return outputs
        except Exception as e:
            logger.error(f"LỖI TRONG QUÁ TRÌNH SCROLL: {e}")
            return []

    # ==================================================
    # ================== HYBRID SEARCH =================
    def hybrid_search(
        self,
        query: str,
        top_k: int = 6,
        alpha: float = 0.8,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """
        Thực hiện tìm kiếm kết hợp (Hybrid Search) giữa vector embedding và từ khóa.

        Args:
            query (str): Câu truy vấn.
            top_k (int, optional): Số lượng kết quả trả về. Mặc định = 3.
            alpha (float, optional): Trọng số cho điểm ngữ nghĩa (0.0 - 1.0).
            filter_payload (dict, optional): Điều kiện lọc payload.

        Returns:
            List[dict]: Danh sách kết quả sắp xếp theo điểm hybrid (ngữ nghĩa + từ khóa).
        """
        try:
            # 1. Tiền xử lý tiếng Việt
            query_clean = preprocess_vi_for_jina(query)

            # 2. Sinh vector embedding cho câu truy vấn
            vector_query = embedding_model.embed_query(query_clean)

            # 3. Thiết lập bộ lọc nếu có
            q_filter = None
            if filter_payload:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
                q_filter = models.Filter(must=must_conditions)

            # 4. Tìm kiếm vector (ngữ nghĩa)
            vector_results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector_query,
                query_filter=q_filter,
                limit=top_k * 2,  # lấy nhiều hơn để trộn kết quả từ khóa
            )

            # 5. Tính điểm kết hợp (semantic + keyword)
            combined = {}
            for point in vector_results.points:
                payload = point.payload or {}
                text = payload.get("text", "")
                keyword_score = 1.0 if query_clean.lower() in text.lower() else 0.0
                semantic_score = point.score or 0.0
                fused_score = alpha * semantic_score + (1 - alpha) * keyword_score

                combined[point.id] = {
                    "text": text,
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "score": fused_score,
                }

            # 6. Sắp xếp kết quả theo điểm tổng hợp
            sorted_results = sorted(
                combined.values(), key=lambda x: x["score"], reverse=True
            )

            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"LỖI TRONG QUÁ TRÌNH HYBRID SEARCH: {e}")
            return []


if __name__ == "__main__":
    user_query = "Hãy cho tôi biết encoder, decoder là gì vậy?"
    logger.info("=== THỰC HIỆN HYBRID SEARCH (SEMANTIC + KEYWORD) ===")

    searcher = QdrantSearcher()
    hybrid_results = searcher.hybrid_search(
        query=user_query,
        top_k=3,
        alpha=0.6,  # 60% ngữ nghĩa + 40% từ khóa
        filter_payload={"source": "bao_cao_image_captioning.pdf"},
    )

    for i, item in enumerate(hybrid_results, start=1):
        print(f"[{i}] {item['text'][:100]}...")
        print(f"    ➤ Score: {item['score']:.4f}")
        print(f"    ➤ Source: {item.get('source', 'N/A')}\n")

    print("==================== SEMANTIC SEARCH ========================")

    basic_results = searcher.semantic_search(
        query=user_query,
        top_k=3,
        use_exact=False,
        hnsw_ef_construct=64,
        filter_payload={"source": "bao_cao_image_captioning.pdf"},
    )

    for i, item in enumerate(basic_results, start=1):
        print(f"[{i}] {item['text'][:100]}...")
        print(f"    ➤ Score: {item['score']:.4f}")
        print(f"    ➤ Source: {item.get('source', 'N/A')}\n")
