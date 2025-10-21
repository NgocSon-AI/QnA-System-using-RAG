# src/vector_db/searcher.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.embedding.embedding import ModelEmbeddings
from src.vector_db.client import QdrantIngestor
from src.utils.logger import Logger
from src.utils.text_cleaner import TextCleaner


class QdrantSearcher:
    """
    Thực hiện các chiến lược tìm kiếm: semantic, keyword và hybrid.
    """

    def __init__(
        self,
        embedding_model: ModelEmbeddings,
        qdrant_db: QdrantIngestor,
        collection_name: str,
        text_cleaner: TextCleaner,
        log_name: str = "QdrantSearcher",
    ) -> None:
        self.logger = Logger(name=log_name).get_logger()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.qdrant_db = qdrant_db
        self.text_cleaner = text_cleaner

        # Load corpus, point_ids và payload từ Qdrant
        self.corpus, self.point_ids, self.corpus_payloads = (
            self._load_corpus_from_qdrant()
        )
        self.tokenized_corpus = [
            self.text_cleaner.clean(doc).split() for doc in self.corpus
        ]

        # BM25
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
            self.logger.warning("⚠️ Corpus rỗng, BM25 sẽ không hoạt động.")

    # ==========================================================
    # 🔹 Load corpus từ Qdrant
    # ==========================================================
    def _load_corpus_from_qdrant(
        self,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Lấy toàn bộ text + point_id + payload từ Qdrant để BM25 + hybrid search."""
        scroll_result = self.qdrant_db.client.scroll(
            collection_name=self.collection_name, limit=5000, with_payload=True
        )
        all_docs = scroll_result[0]

        corpus = [p.payload.get("text", "") for p in all_docs]
        point_ids = [p.id for p in all_docs]
        payloads = [p.payload for p in all_docs]
        return corpus, point_ids, payloads

    # ==========================================================
    # 🔹 Semantic Search
    # ==========================================================
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        with_payload: bool = True,
        filter_payload: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search sử dụng Qdrant."""
        try:
            query_vector = self.embedding_model.embed_query(query)

            qdrant_filter = None
            if filter_payload:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(key=k, match=MatchValue(value=v))
                        for k, v in filter_payload.items()
                    ]
                )

            hits = self.qdrant_db.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=with_payload,
                query_filter=qdrant_filter,
            )

            # hits.result chứa ScoredPoint
            results = [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in hits.points
            ]
            self.logger.info(f"✅ Semantic search: '{query}' → {len(results)} results")
            return results

        except Exception as e:
            self.logger.exception(f"❌ Semantic search error: {e}")
            raise

    # ==========================================================
    # 🔹 Keyword Search (BM25)
    # ==========================================================
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Keyword search sử dụng BM25."""
        if not self.bm25:
            self.logger.warning("⚠️ BM25 corpus rỗng, trả về danh sách rỗng.")
            return []

        try:
            cleaned_query = self.text_cleaner.clean(query)
            tokenized_query = cleaned_query.split()

            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = [
                {
                    "id": self.point_ids[i],
                    "score": float(scores[i]),
                    "payload": self.corpus_payloads[i],
                }
                for i in top_indices
            ]

            self.logger.info(f"✅ Keyword search: '{query}' → {len(results)} results")
            return results

        except Exception as e:
            self.logger.exception(f"❌ Keyword search error: {e}")
            raise

    # ==========================================================
    # 🔹 Hybrid Search
    # ==========================================================
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Kết hợp semantic + keyword search.
        alpha = trọng số semantic, 1-alpha = trọng số keyword
        """
        try:
            sem_results = self.semantic_search(query, top_k=top_k)
            kw_results = self.keyword_search(query, top_k=top_k)

            sem_scores = {r["id"]: r["score"] for r in sem_results}
            kw_scores = {r["id"]: r["score"] for r in kw_results}

            all_ids = set(sem_scores) | set(kw_scores)
            combined_results = []
            for pid in all_ids:
                combined_score = alpha * sem_scores.get(pid, 0.0) + (
                    1 - alpha
                ) * kw_scores.get(pid, 0.0)

                # Chọn payload ưu tiên semantic, fallback BM25
                payload = None
                if pid in sem_scores:
                    payload = next(r["payload"] for r in sem_results if r["id"] == pid)
                elif pid in kw_scores:
                    payload = next(r["payload"] for r in kw_results if r["id"] == pid)

                combined_results.append(
                    {"id": pid, "score": combined_score, "payload": payload}
                )

            combined_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = combined_results[:top_k]

            self.logger.info(
                f"✅ Hybrid search: '{query}' → {len(final_results)} results"
            )
            return final_results

        except Exception as e:
            self.logger.exception(f"❌ Hybrid search error: {e}")
            raise
