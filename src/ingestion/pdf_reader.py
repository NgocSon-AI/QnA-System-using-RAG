from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from src.utils.logger import Logger


class PDFReader:
    """
    Class đọc PDF và xuất ra list các page text.
    """

    def __init__(self, log_name: str = "PDFReader"):
        """_summary_

        Args:
            log_name (str, optional): _description_. Defaults to "PDFReader".
        """
        self.logger = Logger(name=log_name).get_logger()

    def read_pdf(self, file_path: str) -> List[str]:
        """
        Đọc PDF và trả về list các trang text.

        Args:
            file_path (str): đường dẫn file PDF

        Returns:
            List[str]: list các trang text
        """
        file_path = Path(file_path)
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            self.logger.error(f"File PDF không tồn tại hoặc không hợp lệ: {file_path}")
            return []

        self.logger.info(f"Đang đọc file PDF: {file_path}")
        pages_text = []

        try:
            with fitz.open(file_path) as pdf:
                self.logger.info(f"Số trang trong PDF: {pdf.page_count}")
                for i, page in enumerate(pdf):
                    text = page.get_text()
                    pages_text.append(text)
                    self.logger.debug(f"Đã đọc trang {i+1} / {pdf.page_count}")
        except Exception as e:
            self.logger.exception(f"Lỗi khi đọc PDF: {file_path} - {e}")

        self.logger.info(f"Đọc xong file PDF: {file_path}")
        return pages_text


# 🔹 Test nhanh khi chạy trực tiếp
if __name__ == "__main__":
    reader = PDFReader()
    pdf_file = "./data/raw/bao_cao_ imagecaptioning.pdf"
    texts = reader.read_pdf(pdf_file)
    for i, page_text in enumerate(texts):
        print(f"--- Page {i+1} ---")
        print(page_text[:200])  # chỉ show 200 ký tự đầu
