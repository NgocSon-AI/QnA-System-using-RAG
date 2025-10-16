from typing import List
from src.utils.logger import Logger
import re


class TextSplitter:
    """
    Chia text thành chunk để embedding.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        log_name: str = "TextSplitter",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = Logger(name=log_name).get_logger()

    def split_text(self, text: str) -> List[str]:
        """
        Chia 1 text thành chunk
        """
        if not text:
            self.logger.warning("Text rỗng, không thể chia chunk")
            return []

        self.logger.debug(
            f"Chia text dài {len(text)} ký tự thành chunk size={self.chunk_size} với overlap={self.chunk_overlap}"
        )

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            self.logger.debug(
                f"Chunk {len(chunks)}: start={start}, end={end}, len={len(chunk)}"
            )
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def split_pages(self, pages: List[str]) -> List[str]:
        """
        Nhận list các trang PDF, trả về list chunk từ tất cả các trang
        """
        all_chunks = []
        for i, page_text in enumerate(pages):
            self.logger.info(f"Chia page {i+1} dài {len(page_text)} ký tự")
            page_chunks = self.split_text(page_text)
            all_chunks.extend(page_chunks)
        self.logger.info(f"Tổng số chunk sau khi chia tất cả pages: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    from src.ingestion.pdf_reader import PDFReader

    pdf_file = "data/raw/bao_cao_ imagecaptioning.pdf"

    # 1️⃣ đọc PDF -> list trang
    reader = PDFReader()
    pages = reader.read_pdf(pdf_file)

    # 2️⃣ chia mỗi trang thành chunk
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_pages(pages)

    print(f"Tổng chunk tạo ra: {len(chunks)}")
    print("Chunk đầu tiên:", chunks[0][:200])
