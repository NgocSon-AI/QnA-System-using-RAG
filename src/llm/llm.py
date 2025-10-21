# src/llm/llm_generator.py
from typing import List, Dict, Any
from dataclasses import dataclass
from groq import Groq

from src.utils.config import get_settings
from src.utils.logger import Logger


@dataclass
class LLMConfig:
    """
    Cấu hình cho Large Language Model (LLM).
    """

    model_name: str
    temperature: float
    max_tokens: int
    api_key: str

    @classmethod
    def from_settings(cls) -> "LLMConfig":
        """
        Tạo cấu hình LLM từ file settings.
        """
        settings = get_settings()
        return cls(
            model_name=settings.GROQ_MODEL_NAME,
            temperature=0.3,
            max_tokens=1024,
            api_key=settings.GROQ_API_KEY,
        )


class LLMGenerator:
    """
    Sinh câu trả lời từ LLM dựa trên ngữ cảnh được cung cấp.
    """

    def __init__(self, config: LLMConfig, log_name: str = "LLMGenerator") -> None:
        self.config = config
        # Initialize client only if api_key present
        self.client = None
        if config.api_key:
            try:
                self.client = Groq(api_key=config.api_key)
            except Exception:
                self.client = None
        self.logger = Logger(name=log_name).get_logger()
        self.logger = Logger(name=log_name).get_logger()

    # ==========================================================
    # 🔹 Xây dựng prompt
    # ==========================================================
    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Tạo prompt đầu vào cho LLM dựa trên câu hỏi và các đoạn ngữ cảnh.

        Args:
            query (str): Câu hỏi của người dùng.
            contexts (List[Dict]): Danh sách ngữ cảnh từ Qdrant (có payload.text).

        Returns:
            str: Prompt hoàn chỉnh gửi lên LLM.
        """
        context_text = "\n\n".join(
            [
                c["payload"]["text"]
                for c in contexts
                if isinstance(c, dict) and "payload" in c and "text" in c["payload"]
            ]
        )

        prompt = f"""
Bạn là một trợ lý AI trả lời câu hỏi.
Nhiệm vụ của bạn là đọc kỹ bối cảnh được cung cấp dưới đây và trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn.

**QUY TẮC BẮT BUỘC:**
1. CHỈ được phép sử dụng thông tin từ mục [Bối cảnh từ tài liệu] để hình thành câu trả lời.
2. KHÔNG được sử dụng kiến thức bên ngoài hoặc thông tin có sẵn trong mô hình của bạn.
3. Nếu câu trả lời không có trong bối cảnh, hãy trả lời thẳng thắn là: "Tôi không tìm thấy thông tin để trả lời câu hỏi này trong tài liệu được cung cấp."

[Bối cảnh từ tài liệu]:
{context_text}

[Câu hỏi của người dùng]:
{query}

[Câu trả lời của bạn]:
"""
        return prompt.strip()

    # ==========================================================
    # 🔹 Sinh câu trả lời
    # ==========================================================
    def generate_answer(
        self, query: str, contexts: List[Dict[str, Any]], debug: bool = False
    ) -> Dict[str, Any]:
        """
        Gọi LLM để sinh câu trả lời từ truy vấn và ngữ cảnh.

        Args:
            query (str): Câu hỏi của người dùng.
            contexts (List[Dict]): Danh sách ngữ cảnh (từ Qdrant search).
            debug (bool, optional): In prompt để debug. Mặc định False.

        Returns:
            Dict[str, Any]: Kết quả gồm query, answer và context_used.
        """
        prompt = self.build_prompt(query, contexts)

        if debug:
            self.logger.info(f"🧠 Prompt gửi lên LLM:\n{prompt}\n")

        try:
            if not self.client:
                raise RuntimeError(
                    "LLM client not initialized (missing or invalid API key)"
                )

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI thông minh và lịch sự.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            self.logger.info("✅ LLM trả lời thành công.")
        except Exception as e:
            self.logger.exception("❌ Lỗi khi gọi LLM:")
            answer = f"[Lỗi khi gọi LLM]: {str(e)}"

        # If API returned empty string or None, provide a safe fallback
        if not answer:
            self.logger.warning("⚠️ LLM trả lời rỗng, trả fallback message.")
            answer = "Tôi không tìm thấy thông tin để trả lời câu hỏi này trong tài liệu được cung cấp."

        return {
            "query": query,
            "answer": answer,
            "context_used": contexts,
        }
