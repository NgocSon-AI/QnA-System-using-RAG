from qdrant_client import QdrantClient
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff
from config import get_settings
import logging

# Cấu hình logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def hnsw_indexing_updated(collection_name: str, host: str, port: int) -> None:
    """
    Cập nhật cấu hình HNSW indexing cho collection trong Qdrant.

    Mục đích:
        Tối ưu hóa tốc độ truy vấn vector bằng cách điều chỉnh thông số HNSW graph
        và cấu hình bộ tối ưu (optimizer) trong Qdrant.

    Tham số:
        collection_name (str): Tên collection cần cập nhật.
        host (str): Địa chỉ Qdrant server.
        port (int): Cổng kết nối Qdrant server.

    Trả về:
        None
    """
    try:
        client = QdrantClient(host=host, port=port)

        # Kiểm tra collection có tồn tại hay chưa
        collections = [col.name for col in client.get_collections().collections]
        if collection_name not in collections:
            logger.error(f"Collection '{collection_name}' không tồn tại trên Qdrant.")
            return

        # Cấu hình HNSW
        hnsw_config = HnswConfigDiff(
            m=16,  # Số liên kết cho mỗi node trong graph
            ef_construct=64,  # Số lượng neighbors xem xét khi xây dựng index
            full_scan_threshold=80,  # Số điểm nhỏ hơn ngưỡng này thì full scan
            max_indexing_threads=4,  # Giới hạn số luồng indexing song song
            on_disk=False,  # Lưu HNSW index trên RAM để truy cập nhanh hơn
        )

        # Cấu hình optimizer (tối ưu quá trình indexing)
        optimizer_config = OptimizersConfigDiff(
            indexing_threshold=0,  # luôn bật indexing ngay cả với dữ liệu nhỏ
        )

        # Thực hiện cập nhật
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=hnsw_config,
            optimizers_config=optimizer_config,
        )

        logger.info(
            f"✅ Đã tối ưu HNSW indexing cho collection '{collection_name}' thành công!"
        )

    except Exception as e:
        logger.exception(
            f"⚠️ Lỗi khi cập nhật HNSW indexing cho '{collection_name}': {e}"
        )


if __name__ == "__main__":
    hnsw_indexing_updated(
        settings.COLLECTION_NAME,
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
