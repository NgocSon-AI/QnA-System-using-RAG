import re


def preprocess_vi_for_jina(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt trước khi đưa vào mô hình embedding (ví dụ: Jina Embeddings).

    Hàm này thực hiện các bước chuẩn hóa:
      - Loại bỏ khoảng trắng ở đầu và cuối chuỗi.
      - Ép kiểu sang chuỗi nếu đầu vào không phải là str (ví dụ: int, float,...).
      - Thay thế các khoảng trắng liên tiếp bằng một khoảng trắng duy nhất.
    Tham số:
        text (str): Văn bản đầu vào cần tiền xử lý. Có thể là chuỗi hoặc kiểu dữ liệu khác.

    Trả về:
        str: Chuỗi văn bản đã được làm sạch và chuẩn hóa.
    """

    # Nếu là chuỗi thì loại bỏ khoảng trắng đầu/cuối
    if isinstance(text, str):
        text = text.strip()
    else:
        # Ép kiểu sang chuỗi nếu đầu vào là số hoặc kiểu dữ liệu khác
        text = str(text).strip()

    # Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text


if __name__ == "__main__":
    text = "Tiền xử lý query:     loại bỏ whitespace thừa, chuẩn hóa text.    "
    print("Kết quả sau tiền xử lý:", preprocess_vi_for_jina(text))
