from functools import lru_cache
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Load toàn bộ biến môi trường từ file .env
load_dotenv()


@dataclass
class Settings:
    # GROQ / LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # PDF
    PDF_PATH: str = "data/raw/bao_cao_imagecaptioning.pdf"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDARNT_DISTANCE: str = "cosine"
    COLLECTION_NAME: str = "pdf_documents"
    VECTOR_SIZE: int = 768

    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # Embedding / device
    JINA_MODEL_NAME: str = "Alibaba-NLP/gte-multilingual-base"
    JINA_TASK: str = "retrieval.passage"
    MODEL_TOKEN_NAME: str = "text-embedding-3-small"
    DEVICE: str = "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
