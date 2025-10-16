from datetime import datetime
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(
        self,
        name: str = "RAGLogger",
        log_dir: str = "./logs",
        level_console=logging.INFO,
        level_file=logging.DEBUG,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Tạo tên file log có timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] - [%(name)s] - %(levelname)s : %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_console)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / log_file, encoding="utf-8")
        file_handler.setLevel(level_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


# 🔹 Example usage
if __name__ == "__main__":
    my_logger = Logger(name="TestLogger").get_logger()
    my_logger.debug("Debug message")
    my_logger.info("Info message")
    my_logger.warning("Warning message")
    my_logger.error("Error message")
