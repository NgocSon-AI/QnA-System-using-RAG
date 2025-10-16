from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )
    # ==================== GROQ CONFIG ====================
    GROQ_API_KEY: str
    GROQ_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    PDF_PATH: str = "bao_cao_ imagecaptioning.pdf"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "pdf_documents"

    CHUNK_SIZE: int = 500  # Số ký tự mỗi chunk
    CHUNK_OVERLAP: int = 100  # Số ký tự overlap giữa các chunk

    JINA_MODEL_NAME: str = "Alibaba-NLP/gte-multilingual-base"
    JINA_TASK: str = "retrieval.passage"

    DEVICE: str = "cpu"  # Đổi thành "cuda" nếu có GPU

    # =====================================================


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
