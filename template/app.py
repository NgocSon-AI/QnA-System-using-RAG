# app.py
import streamlit as st
from embedding_query_and_searches_ngocson import QdrantSearcher
from groq import Groq
from config import get_settings

# ---------------------------------------------------------------------------
# ⚙️ Load cấu hình và khởi tạo client
# ---------------------------------------------------------------------------
settings = get_settings()  # Load các thiết lập từ config.py

# Khởi tạo Groq LLM client
groq_client = Groq(api_key=settings.GROQ_API_KEY)

# Khởi tạo Qdrant searcher để thực hiện các loại search
searcher = QdrantSearcher()


# ---------------------------------------------------------------------------
# 🖥️ Giao diện người dùng Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid Search + LLM", layout="wide")

st.title("🤖 Hybrid Search + Groq LLM Assistant")
st.write("Nhập câu truy vấn để tìm kiếm dữ liệu và sinh câu trả lời thông minh.")

# Input query từ người dùng
query = st.text_area(
    "Nhập truy vấn:", placeholder="Ví dụ: Transformer là gì?", height=100
)

# Lựa chọn chiến lược tìm kiếm
search_mode = st.radio(
    "Chọn chiến lược tìm kiếm:",
    ["Hybrid Search", "Semantic Search"],
    horizontal=True,
)

# Slider cho top_k và alpha
col1, col2 = st.columns(2)

with col1:
    top_k = st.slider("Số kết quả hiển thị:", 1, 10, 6)

with col2:
    # 🔹 Alpha chỉ khả dụng khi chọn Hybrid Search
    if search_mode == "Hybrid Search":
        alpha = st.slider(
            "Hệ số alpha (0 = keyword, 1 = semantic):", 0.0, 1.0, 0.6, step=0.1
        )
    else:
        # Disabled slider, chỉ hiển thị giá trị mặc định
        alpha = st.slider(
            "Hệ số alpha (chỉ áp dụng cho Hybrid Search):",
            0.0,
            1.0,
            0.6,
            step=0.1,
            disabled=True,
        )


# ---------------------------------------------------------------------------
# 🔍 Xử lý truy vấn khi người dùng nhấn nút "Hỏi AI"
# ---------------------------------------------------------------------------
if st.button("🔍 Hỏi AI"):
    if not query.strip():
        st.warning("⚠️ Vui lòng nhập nội dung truy vấn.")
    else:
        with st.spinner("🔎 Đang tìm kiếm dữ liệu..."):
            try:
                # Chọn loại search phù hợp
                if search_mode == "Hybrid Search":
                    results = searcher.hybrid_search(query, top_k=top_k, alpha=alpha)
                elif search_mode == "Semantic Search":
                    results = searcher.semantic_search(query, top_k=top_k)
            except Exception as e:
                st.error(f"Lỗi khi gọi hàm search: {e}")
                results = []

        # -------------------------------------------------------------------
        # Hiển thị kết quả tìm kiếm từ Qdrant
        # -------------------------------------------------------------------
        if results:
            with st.expander(
                f"🔍 Hiển thị {len(results)} kết quả tìm kiếm", expanded=False
            ):
                st.success(f"Đã tìm thấy {len(results)} kết quả từ Qdrant:")
                for i, res in enumerate(results, start=1):
                    st.markdown(f"**{i}. {res.get('text', '')}**")
                    st.caption(
                        f"Score: {res.get('score', 0):.4f} | Source: {res.get('source', '')}"
                    )

            # -------------------------------------------------------------------
            # Chuẩn bị bối cảnh (context) cho LLM
            # -------------------------------------------------------------------
            context = "\n".join([r.get("text", "") for r in results])

            # Tạo prompt chi tiết cho AI, đảm bảo chỉ sử dụng thông tin từ Qdrant
            prompt = f"""
Bạn là một trợ lý AI trả lời câu hỏi.
Nhiệm vụ của bạn là đọc kỹ bối cảnh được cung cấp dưới đây và trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn.

**QUY TẮC BẮT BUỘC:**
1. CHỈ được phép sử dụng thông tin từ mục [Bối cảnh từ tài liệu] để hình thành câu trả lời.
2. KHÔNG được sử dụng kiến thức bên ngoài hoặc thông tin có sẵn trong mô hình của bạn.
3. Nếu câu trả lời không có trong bối cảnh, hãy trả lời thẳng thắn là: "Tôi không tìm thấy thông tin để trả lời câu hỏi này trong tài liệu được cung cấp."

[Bối cảnh từ tài liệu]:
{context}

[Câu hỏi của người dùng]:
{query}
[Câu trả lời của bạn]:
"""

            # -------------------------------------------------------------------
            # Gọi Groq API để sinh câu trả lời
            # -------------------------------------------------------------------
            with st.spinner("💭 Đang sinh câu trả lời từ Groq..."):
                try:
                    response = groq_client.chat.completions.create(
                        model=settings.GROQ_MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "Bạn là trợ lý AI thân thiện và hữu ích.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.5,
                    )
                    answer = response.choices[0].message.content
                    st.markdown("### 💬 Phản hồi từ AI:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Lỗi khi gọi Groq API: {e}")

        else:
            st.warning("❌ Không tìm thấy kết quả nào trong Qdrant.")
