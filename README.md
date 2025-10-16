# QnA-System-using-RAG

Created By NgocSonNguyen

# AI-RAG Project with Qdrant

- Đây là project **RAG (Retrieval-Augmented Generation)** sử dụng **Qdrant** làm vector database, **Jina embeddings** cho semantic search, và **Streamlit** để tương tác.

- Đầu vào sẽ là tài liệu file.pdf + các query từ người dùng và đầu ra của mô hình là câu trả lời cho câu hỏi dựa trên tài liệu đã được cung cấp

## Ghi chú chi tiết

### **Cấu trúc thư mục**

```bash
├─ main.py                                      # Pipeline tạo embedding tinh chỉnh tham số hnsw, thử nghiệm chiến lược tìm kiếm
├─ app.py                                       # Streamlit UI
├─ config.py                                    # Config project
├─ preprocess_fn_basic_ngocson.py               # Hàm tiền xử lý văn bản, loại bỏ khoảng trắng
├─ hnsw_indexing_update.py                      # Tinh chỉnh một số tham số trong việc tạo index
├─ embedding_query_and_searches_ngocson.py      # Hàm embedding và các method search
├─ script_test.py                               # Load PDF, chia thành các chunk, gọi model embedding chuyển các chunk thành embedding và lưu vào Qdrant vector database
├─ requirements.txt                             # quản lý thư viện (dependencies)
└─ README.md
```

---

## 🚀 Hướng dẫn cài đặt và chạy project

### **Bước 1: Clone repository**

```bash
git clone git@gitlab.com:enso.ai/tts/qdrant_rnd.git
```

### **Bước 2: Di chuyển vào thư mục project**

```bash

cd qdrant_rnd
```

### **Bước 3: Tạo môi trường ảo với Conda**

```bash

conda create -n qdrant python==3.12
```

### **Bước 4: Cài đặt các thư viện**

```bash

# Kích hoạt môi trường
conda activate qdrant   # Linux / macOS
# conda activate qdrant # Windows
# Cài đặt dependencies
pip install -r requirements.txt
```

### **Bước 5: Tạo file .env**

```bash
# 1. Trên Linux / macOS
echo "GROQ=your_groq_api_key" > .env
# 2. Trên Windows (CMD truyền thống)
echo GROQ=your_groq_api_key > .env
```

### **Bước 6: Chạy Docker Compose**

```bash
# Hãy mở docker desktop trước
docker compose up -d
# Lưu ý: Docker sẽ chạy Qdrant local và các service cần thiết.
```

### **Bước 7: Chạy pipeline tạo embedding**

```bash
python3 main.py
```

### **Bước 8: Chạy Streamlit UI**

```bash
streamlit run app.py
#Mở trình duyệt theo link hiển thị để trải nghiệm giao diện tương tác với RAG + LLM.
```
