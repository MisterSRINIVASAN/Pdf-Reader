# 📄 PDF Chatbot using Streamlit & LangChain

An interactive AI-powered chatbot that allows users to upload a PDF and ask questions about its content. The application processes the document, creates embeddings, and retrieves relevant answers using a Large Language Model.

---

## 🚀 Features

- 📂 Upload any PDF document
- 💬 Ask questions in natural language
- ⚡ Fast retrieval using FAISS vector store
- 🧠 Context-aware answers using Hugging Face LLM
- 🔄 Chat history maintained using Streamlit session state
- ⏳ Efficient processing (PDF is processed only once)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **LLM:** Hugging Face (FLAN-T5)  
- **Embeddings:** HuggingFaceEmbeddings  
- **Vector DB:** FAISS (Facebook AI Similarity Search)  
- **PDF Processing:** PyPDF2  
- **Framework:** LangChain  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/MisterSRINIVASAN/Pdf-Reader.git
cd pdf-reader
