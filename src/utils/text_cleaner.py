import re
from src.utils.logger import Logger


class TextCleaner:
    """
    Dọn dẹp text, loại bỏ ký tự đặc biệt, khoảng trắng thừa, v.v.
    """

    def __init__(self, log_name: str = "TextCleaner"):
        self.logger = Logger(name=log_name).get_logger()

    @staticmethod
    def clean(text: str) -> str:
        """
        Args:
            text (str): text cần làm sạch
        Returns:
            str: text đã làm sạch
        """
        if not text:
            return ""

        # 1 Remove nhiều khoảng trắng
        text = re.sub(r"\s+", " ", text)
        # 2 Remove số trang dạng "Page 1/10"
        text = re.sub(r"Page \d+/\d+", "", text, flags=re.IGNORECASE)
        # 3 Strip đầu cuối
        return text.strip()
