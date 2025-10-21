from pathlib import Path
from typing import List

from src.utils.logger import Logger


try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import PyPDF2

    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False


class PDFReader:
    """Read PDF and return a list of page texts.

    The reader will try PyMuPDF first (faster and more reliable), and fall back
    to PyPDF2 if PyMuPDF is not available or fails for a specific file.
    """

    def __init__(self, log_name: str = "PDFReader") -> None:
        self.logger = Logger(name=log_name).get_logger()

    def read_pdf(self, file_path: str) -> List[str]:
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".pdf":
            self.logger.error("File PDF không tồn tại hoặc không hợp lệ: %s", file_path)
            return []

        # Try PyMuPDF first
        if _HAS_FITZ:
            try:
                self.logger.info("Đang đọc PDF với PyMuPDF: %s", file_path)
                pages: List[str] = []
                with fitz.open(path) as pdf:
                    for p in pdf:
                        try:
                            pages.append(p.get_text() or "")
                        except Exception:
                            pages.append("")
                return pages
            except Exception as e:
                self.logger.warning("PyMuPDF failed (%s), falling back to PyPDF2", e)

        # Fallback to PyPDF2
        if _HAS_PYPDF2:
            try:
                self.logger.info("Đang đọc PDF với PyPDF2: %s", file_path)
                pages = []
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                return pages
            except Exception as e:
                self.logger.exception("PyPDF2 failed to read PDF: %s", e)

        self.logger.error("No PDF reader available (install pymupdf or pypdf2)")
        return []


if __name__ == "__main__":
    reader = PDFReader()
    pdf_file = "data/raw/bao_cao_ imagecaptioning.pdf"
    texts = reader.read_pdf(pdf_file)

    for i, page_text in enumerate(texts):
        print(f"\n--- PAGE {i + 1} ---")
        print(page_text[:200].strip() or "[Trang trống]")
