import os
import warnings
from pathlib import Path
from typing import List
import PyPDF2
import torch
from transformers import AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from preprocess_fn_basic_ngocson import preprocess_vi_for_jina
from config import get_settings


settings = get_settings()


from sentence_transformers import SentenceTransformer
from preprocess_fn_basic_ngocson import preprocess_vi_for_jina


from sentence_transformers import SentenceTransformer
from preprocess_fn_basic_ngocson import preprocess_vi_for_jina


class JinaEmbeddings:
    def __init__(
        self, model_name: str, task: str = "retrieval.passage", device: str = "cpu"
    ):
        """
        Jina embedding wrapper sử dụng SentenceTransformer cho Alibaba-NLP/gte-multilingual-base.
        """
        self.model_name = model_name
        self.task = task
        self.device = device

        print(f"🔧 Loading Jina model: {model_name}")
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True  # 👈 thêm dòng này
        )
        print(f"Model loaded on device: {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [preprocess_vi_for_jina(t) for t in texts]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_tensor=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = preprocess_vi_for_jina(text)
        return self.embed_documents([text])[0]

    def close(self):
        if hasattr(self, "model"):
            del self.model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_embedding_model():
    """Tạo và trả về Jina embedding model"""
    return JinaEmbeddings(
        model_name=settings.JINA_MODEL_NAME,
        task=settings.JINA_TASK,
        device=settings.DEVICE,
    )


def load_pdf(pdf_path: str) -> str:
    """Load và extract text từ PDF"""
    print(f"Loading PDF from: {pdf_path}")

    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"   Found {num_pages} pages")

        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text += page_text + "\n"
            print(f"   Extracted page {page_num}/{num_pages}")

    print(f"Total characters extracted: {len(text)}")
    return text


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """Chia text thành các chunks với overlap"""
    print(f"\nChunking text (size={chunk_size}, overlap={overlap})")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Loại bỏ whitespace thừa
        text = preprocess_vi_for_jina(text)

        if chunk:  # Chỉ thêm chunk không rỗng
            chunks.append(chunk)

        start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks")
    return chunks


def create_embeddings(
    chunks: List[str], embedding_model: JinaEmbeddings
) -> List[List[float]]:
    """Tạo embeddings cho các chunks sử dụng Jina model"""
    print(f"\nCreating embeddings with Jina model")
    print(f"Processing {len(chunks)} chunks...")

    embeddings = embedding_model.embed_documents(chunks)

    print(f"Created {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    return embeddings


def store_in_qdrant(chunks: List[str], embeddings: List[List[float]]):
    """Lưu embeddings vào Qdrant"""
    print(f"\nConnecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    # Kết nối tới Qdrant
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # Lấy vector dimension
    vector_size = len(embeddings[0])

    # Xóa collection cũ nếu tồn tại (để test)
    try:
        client.delete_collection(collection_name=settings.COLLECTION_NAME)
        print(f"   Deleted existing collection: {settings.COLLECTION_NAME}")
    except Exception:
        pass

    # Tạo collection mới
    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection: {settings.COLLECTION_NAME} (dimension: {vector_size})")

    # Chuẩn bị points để upload
    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "chunk_index": idx, "source": settings.PDF_PATH},
        )
        points.append(point)

    # Upload tất cả points
    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)

    print(f"Uploaded {len(points)} vectors to Qdrant")

    # Verify
    collection_info = client.get_collection(collection_name=settings.COLLECTION_NAME)
    print(f"   Collection vector count: {collection_info.vectors_count}")
