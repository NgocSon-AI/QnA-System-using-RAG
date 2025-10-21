from typing import List
from src.utils.logger import Logger
from src.utils.text_cleaner import TextCleaner

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


class MockEmbedder:
    """Fallback embedder dùng khi không có sentence-transformers."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = abs(hash(t))
            vec = [((h >> (j % 64)) & 0xFF) / 255.0 for j in range(self.dim)]
            out.append(vec)
        return out

    def embed_query(self, q: str) -> List[float]:
        return self.embed_documents([q])[0]


class ModelEmbeddings:
    """Wrapper cho SentenceTransformer để tạo embedding cho text hoặc query."""

    _model_cache = None

    def __init__(self, model_name: str, task: str, device: str, log_name: str):
        self.logger = Logger(name=log_name).get_logger()
        self.model_name = model_name
        self.task = task
        self.device = device

        if ModelEmbeddings._model_cache is None:
            if _HAS_SENTENCE_TRANSFORMERS:
                self.logger.info("🚀 Đang load model embedding: %s", model_name)
                try:
                    # ✅ SỬA Ở ĐÂY
                    ModelEmbeddings._model_cache = SentenceTransformer(
                        model_name, trust_remote_code=True, device=device
                    )
                    self.logger.info("✅ Model loaded: %s", model_name)
                except Exception as e:
                    self.logger.exception("❌ Lỗi khi load model %s: %s", model_name, e)
                    ModelEmbeddings._model_cache = MockEmbedder(dim=768)
                    self.logger.warning("⚠️ Dùng MockEmbedder do lỗi khi load model")
            else:
                self.logger.warning(
                    "⚠️ sentence-transformers không khả dụng, dùng MockEmbedder"
                )
                ModelEmbeddings._model_cache = MockEmbedder(dim=768)

        self.model = ModelEmbeddings._model_cache

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sinh embedding cho danh sách đoạn văn bản."""
        if not texts:
            self.logger.warning("⚠️ Danh sách text rỗng, không thể tạo embedding.")
            return []

        cleaner = TextCleaner()

        # ✅ SỬA Ở ĐÂY — trích xuất 'text' nếu là dict
        cleaned_texts = [
            cleaner.clean(t["text"] if isinstance(t, dict) else t) for t in texts
        ]

        self.logger.info("🔹Tạo embedding cho %d đoạn văn bản.", len(cleaned_texts))

        if _HAS_SENTENCE_TRANSFORMERS and hasattr(self.model, "encode"):
            embeddings = self.model.encode(
                cleaned_texts,
                normalize_embeddings=True,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        return self.model.embed_documents(cleaned_texts)

    def embed_query(self, query: str) -> List[float]:
        """Sinh embedding cho truy vấn đơn."""
        if not query:
            self.logger.warning("⚠️ Query rỗng, không thể tạo embedding.")
            return []

        cleaner = TextCleaner()
        cleaned_query = cleaner.clean(query)

        if _HAS_SENTENCE_TRANSFORMERS and hasattr(self.model, "encode"):
            embedding = self.model.encode(
                [cleaned_query],
                normalize_embeddings=True,
                convert_to_tensor=False,
            )[0]
            return embedding.tolist()

        return self.model.embed_query(cleaned_query)


def get_embedding_model(
    model_name: str, task: str = "retrieval.passage", device: str = "cpu"
) -> ModelEmbeddings:
    """Factory function tạo instance của ModelEmbeddings."""
    return ModelEmbeddings(
        model_name=model_name, task=task, device=device, log_name="EMBEDDING"
    )
