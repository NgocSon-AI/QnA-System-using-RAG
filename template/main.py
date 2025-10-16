from script_test import (
    load_pdf,
    chunk_text,
    get_embedding_model,
    create_embeddings,
    store_in_qdrant,
)
from config import get_settings
from embedding_query_and_searches_ngocson import QdrantSearcher

settings = get_settings()


# ---------------------------------------------------------------------------
# HÀM HỖ TRỢ NỘI BỘ
# ---------------------------------------------------------------------------


def _get_callable(module_or_obj, *names):
    """
    Hàm tiện ích giúp tìm và lấy về một hàm có thể gọi (callable)
    trong module hoặc đối tượng được truyền vào.

    - Dùng khi có thể tồn tại nhiều tên hàm khác nhau cho cùng chức năng.
    - Tránh gây crash nếu không tìm thấy hàm nào hợp lệ.

    Args:
        module_or_obj: Module hoặc đối tượng cần tìm hàm.
        *names: Danh sách các tên hàm khả dĩ.

    Returns:
        Callable object nếu tìm thấy, ngược lại trả về None.
    """
    for n in names:
        f = getattr(module_or_obj, n, None)
        if callable(f):
            return f
    return None


# ---------------------------------------------------------------------------
# CHƯƠNG TRÌNH CHÍNH
# ---------------------------------------------------------------------------


def main():
    """
    Chạy toàn bộ pipeline xử lý PDF và lưu trữ embedding vào Qdrant.

    Pipeline bao gồm các bước:
        1. Đọc nội dung từ file PDF.
        2. Chia nhỏ văn bản thành các đoạn (chunking).
        3. Khởi tạo mô hình nhúng (embedding model).
        4. Sinh vector embedding cho từng đoạn.
        5. Lưu trữ dữ liệu vào Qdrant vector database.
        6. Cập nhật chỉ mục HNSW để tối ưu truy vấn vector.
    """
    print("=" * 60)
    print("🚀 PDF TO QDRANT EMBEDDING PIPELINE (JINA v3)")
    print("=" * 60)

    try:
        # 1️Đọc dữ liệu từ file PDF
        text = load_pdf(settings.PDF_PATH)

        # 2️Tiền xử lý và chia nhỏ nội dung
        chunks = chunk_text(text)

        # 3️ Khởi tạo mô hình embedding (sử dụng Jina)
        print(f"\n🔧 Initializing Jina embedding model...")
        embedding_model = get_embedding_model()

        # 4️ Sinh vector embedding cho từng đoạn văn bản
        embeddings = create_embeddings(chunks, embedding_model)

        # 5️ Lưu embedding và text vào Qdrant
        store_in_qdrant(chunks, embeddings)

        print("\n" + "=" * 60)
        print("✨ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # 6️ Cập nhật cấu hình chỉ mục HNSW sau khi thêm dữ liệu
        try:
            import hnsw_indexing_update as hnsw_mod

            # Tìm và gọi hàm cập nhật HNSW trong module (nếu có)
            hnsw_func = _get_callable(
                hnsw_mod, "hnsw_indexing_updated", "hnsw_indexing_update"
            )
            if hnsw_func:
                hnsw_func(
                    collection_name=settings.COLLECTION_NAME,
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
            else:
                print(
                    "⚠️ Không tìm thấy hàm cập nhật HNSW trong module hnsw_indexing_update. Bỏ qua bước này."
                )
        except Exception as e:
            print(f"⚠️ Lỗi khi cập nhật HNSW index: {e}")

        # 🧹 Giải phóng bộ nhớ mô hình embedding
        del embedding_model

    except FileNotFoundError:
        print(f"\n❌ Lỗi: Không tìm thấy file tại {settings.PDF_PATH}")
        print("👉 Vui lòng kiểm tra lại đường dẫn PDF_PATH trong file config.")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# CHẠY TRỰC TIẾP MODULE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
