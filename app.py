# app/main.py
import streamlit as st
from qdrant_client import QdrantClient

from src.vector_db.search_strategy import QdrantSearcher
from src.vector_db.client import QdrantIngestor
from src.embedding.embedding import ModelEmbeddings
from src.llm.llm import LLMConfig, LLMGenerator
from src.utils.logger import Logger
from src.utils.config import get_settings
from src.utils.text_cleaner import TextCleaner

# ==============================
# 🔹 Init logger
# ==============================
logger = Logger(name="STREAMLIT_APP").get_logger()
settings = get_settings()
client = QdrantClient(host="localhost", port=6333)  # hoặc "127.0.0.1"

# 2️⃣ Khởi tạo embedding model
embedding_model = ModelEmbeddings(
    model_name="Alibaba-NLP/gte-multilingual-base",
    task="retrieval.passage",
    device="cpu",
    log_name="EMBEDDING",
)

# 3️⃣ Khởi tạo Qdrant ingestor
qdrant_ingestor = QdrantIngestor(
    client=client,
    collection_name=settings.COLLECTION_NAME,
    vector_size=settings.VECTOR_SIZE,
    device=settings.DEVICE,
    log_name="QdrantIngestor",
    reset_collection=False,
)

# 4️⃣ TextCleaner
text_cleaner = TextCleaner("TextCleaner")

# 5️⃣ Searcher
searcher = QdrantSearcher(
    embedding_model=embedding_model,
    qdrant_db=qdrant_ingestor,
    collection_name=settings.COLLECTION_NAME,
    text_cleaner=text_cleaner,
    log_name="SearcherDemo",
)


# ==============================
# 🔹 Init Searcher & LLM
# ==============================
@st.cache_resource(show_spinner=False)
def get_searcher():
    return searcher


@st.cache_resource(show_spinner=False)
def get_llm():
    config = LLMConfig.from_settings()
    return LLMGenerator(config=config)


searcher = get_searcher()
llm = get_llm()

# ==============================
# 🔹 Streamlit UI
# ==============================
st.set_page_config(page_title="RAG QA App", page_icon="🤖", layout="wide")
st.title("RAG QA System 🧠")

st.markdown(
    "Nhập câu hỏi của bạn để RAG pipeline tìm kiếm thông tin trong vector DB "
    "và tạo câu trả lời bằng LLM."
)

# Input user query
user_query = st.text_input("Nhập câu hỏi:", "")

top_k = st.slider("Số kết quả retriever:", min_value=1, max_value=10, value=3)
use_hybrid = st.checkbox("Sử dụng hybrid search", value=True)

if st.button("Gửi câu hỏi") and user_query.strip():
    with st.spinner("🔍 Retrieving context..."):
        try:
            if use_hybrid:
                contexts = searcher.hybrid_search(
                    query=user_query, top_k=top_k, alpha=0.9
                )
            else:
                contexts = searcher.semantic_search(query=user_query, top_k=top_k)

            if not contexts:
                st.warning("Không tìm thấy thông tin phù hợp trong tài liệu!")
            else:
                st.success(f"✅ Tìm thấy {len(contexts)} context(s).")

        except Exception as e:
            st.error(f"❌ Lỗi khi search: {e}")
            contexts = []

    if contexts:
        with st.spinner("💬 Generating answer..."):
            response = llm.generate_answer(
                query=user_query, contexts=contexts, debug=True
            )

            # Hiển thị câu trả lời
            st.markdown("### 🧠 Câu trả lời:")
            st.info(response["answer"])

            # Hiển thị các ngữ cảnh được dùng
            st.markdown("### 📚 Các ngữ cảnh được sử dụng:")

            for i, c in enumerate(contexts, start=1):
                text = c.get("payload", {}).get("text", "Không có nội dung")
                score = c.get("score", None)

                # Tiêu đề của từng ngữ cảnh
                exp_title = f"Context {i}"
                if score is not None:
                    exp_title += f" — 🔢 Score: {score:.4f}"

                # Dùng expander để ẩn/hiện chi tiết ngữ cảnh
                with st.expander(exp_title):
                    st.write(text)
