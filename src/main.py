from src.ingestion.pdf_reader import PDFReader
from src.ingestion.splitter import TextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.vector_db.client import QdrantIngestor
from src.embedding.embedding import get_embedding_model
from src.utils.logger import Logger
from src.utils.config import get_settings
import hashlib


settings = get_settings()

pdf_reader = PDFReader(log_name="PDFReader")
pages = pdf_reader.read_pdf(settings.PDF_PATH)  # -> danh sach cac trang


split_page = TextSplitter(
    settings.CHUNK_SIZE,
    settings.CHUNK_OVERLAP,
    settings.MODEL_TOKEN_NAME,
    log_name="TextSplitter",
)
chunks = split_page.split_pages(pages=pages, source_name=settings.PDF_PATH)


client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

embeddings_model = get_embedding_model(
    model_name=settings.JINA_MODEL_NAME, task=settings.JINA_TASK, device=settings.DEVICE
)
embeddings = embeddings_model.embed_documents(chunks)

ingestor = QdrantIngestor(
    client=client,
    collection_name=settings.COLLECTION_NAME,
    vector_size=settings.VECTOR_SIZE,
    device=settings.DEVICE,
    log_name="QdrantIngestion",
    reset_collection=True,
)
ingestor.upsert_to_qdrant(
    pdf_path=settings.PDF_PATH, chunks=chunks, embeddings=embeddings
)

print("Collection size: ", ingestor.collection_size())
