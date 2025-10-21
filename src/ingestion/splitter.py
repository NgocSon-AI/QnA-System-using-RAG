from typing import List, Dict, Optional

from src.utils.logger import Logger
from src.utils.text_cleaner import TextCleaner
from src.utils.config import get_settings

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False


class TextSplitter:
    """
    Chia text thành các đoạn (chunk) nhỏ để xử lý embedding hoặc lưu vào Vector DB.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
        log_name: str = "TextSplitter",
    ):
        """
        Args:
            chunk_size (int): Số lượng token tối đa mỗi chunk.
            chunk_overlap (int): Số token chồng giữa các chunk.
            model_name (str): Tên model để chọn tokenizer tương ứng.
            log_name (str): Tên logger, mặc định là "TextSplitter".
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = Logger(name=log_name).get_logger()

        if _HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
                self.logger.info(f"✅ Sử dụng tokenizer của model: {model_name}")
            except Exception:
                self.logger.warning(
                    f"⚠️ Model `{model_name}` không được hỗ trợ, dùng `cl100k_base` mặc định."
                )
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback: naive whitespace-based chunking if tiktoken not available
            self.encoding = None
            self.logger.warning(
                "⚠️ tiktoken không có sẵn, sẽ dùng fallback chia theo ký tự (approx)."
            )

    def split_text(
        self, text: str, metadata: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        Chia text thành các chunk kèm metadata.

        Args:
            text (str): Văn bản cần chia.
            metadata (dict): Thông tin kèm theo (VD: source, page).

        Returns:
            List[Dict[str, str]]: Danh sách chunk có payload.
        """
        if not text:
            self.logger.warning("⚠️ Text rỗng, bỏ qua.")
            return []

        chunks = []
        if self.encoding is not None:
            tokens = self.encoding.encode(text)
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                payload = {"text": chunk_text}
                if metadata:
                    payload.update(metadata)
                chunks.append(payload)
                start += self.chunk_size - self.chunk_overlap
        else:
            # Rough character-based fallback (approximate token size)
            approx_char_size = int(self.chunk_size * 4)  # heuristic
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + approx_char_size, text_len)
                chunk_text = text[start:end]
                payload = {"text": chunk_text}
                if metadata:
                    payload.update(metadata)
                chunks.append(payload)
                start += approx_char_size - int(self.chunk_overlap * 4)

        self.logger.debug(f"📄 Chia được {len(chunks)} chunk cho 1 đoạn text.")
        return chunks

    def split_pages(
        self, pages: List[str], source_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Chia nhiều trang PDF thành các chunk, mỗi chunk kèm metadata {text, source, page}.

        Args:
            pages (List[str]): Danh sách các trang văn bản.
            source_name (str): Tên nguồn hoặc file PDF.

        Returns:
            List[Dict[str, str]]: Danh sách tất cả các chunk.
        """
        all_chunks = []
        for i, page_text in enumerate(pages):
            metadata = {"source": source_name, "page": i + 1}
            page_chunks = self.split_text(page_text, metadata)
            all_chunks.extend(page_chunks)

        self.logger.info(f"✅ Tổng số chunk sau khi chia: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    settings = get_settings()
    splitter = TextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        model_name=settings.MODEL_TOKEN_NAME,
    )

    text = "AI là viết tắt của Artificial Intelligence. Đây là một lĩnh vực của khoa học máy tính..."
    chunks = splitter.split_text(text, metadata={"source": "Báo cáo Image Caption"})

    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c['text'][:60]}...\n")
