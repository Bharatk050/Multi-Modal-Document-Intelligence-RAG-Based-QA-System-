# Multi-Modal RAG System

A **Retrieval-Augmented Generation (RAG)** application that processes PDF documents and enables intelligent question-answering using **hybrid search** (BM25 + Vector) and **free, open-source AI models**.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- 📄 **Multi-Modal Document Processing**
  - Text extraction from PDFs
  - Table detection and extraction
  - Image extraction with OCR (Tesseract)

- 🔍 **Hybrid Search**
  - **BM25** (keyword-based) + **FAISS** (semantic vectors)
  - Configurable alpha weight for search blending
  - Pure vector search fallback option

- 🤖 **Free AI Models** (No API keys required!)
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - LLM: `google/flan-t5-small` (with automatic fallback to SimpleQA)

- 💬 **Interactive Chat Interface**
  - Built with Streamlit
  - Chat history with citations
  - Document source tracking

---

## 📁 Project Structure

```
multi-model_assignment/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── document_processor.py   # PDF processing (text, tables, images)
├── process_document.py     # Step 1: Document extraction script
├── create_embeddings.py    # Step 2: Embedding creation script
├── vector_BM25_store.py    # Hybrid vector store (FAISS + BM25)
├── llm_qa.py               # LLM and SimpleQA classes
├── run_pipeline.py         # Run full pipeline automatically
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # Place your PDF here
│   ├── processed/          # Extracted chunks (JSON)
│   ├── images/             # Extracted images
│   └── vector_store/       # FAISS index files
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
env\scripts\activate

# Activate (Linux/Mac)
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image processing)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH or set `TESSERACT_CMD` in your environment

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 3. Add Your PDF Document

Place your PDF file in the `data/raw/` folder:
```
data/raw/qatar_test_doc.pdf
```

Or update `config.py` with your PDF path.

### 4. Process Document & Create Embeddings

**Option A: Run Full Pipeline**
```bash
python run_pipeline.py
```

**Option B: Run Steps Manually**
```bash
# Step 1: Extract text, tables, and images
python process_document.py

# Step 2: Create embeddings and build FAISS index
python create_embeddings.py
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Paths
PDF_PATH = 'data/raw/your_document.pdf'
VECTOR_STORE_PATH = 'data/vector_store/faiss_index'

# Models (free, open-source)
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-small'  # Use 'flan-t5-base' for better quality
```

---

## 🔎 Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Hybrid (BM25 + Vector)** | Combines keyword matching with semantic search | General queries |
| **Vector Only** | Pure semantic similarity search | Conceptual questions |

Toggle between modes using the sidebar in the app.

---

## 💡 How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF Doc   │────▶│  Processor  │────▶│   Chunks    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Embeddings │
                                        │  (MiniLM)   │
                                        └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
             ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
             │    FAISS    │            │    BM25     │            │   Chunks    │
             │   (Vector)  │            │  (Keyword)  │            │   (JSON)    │
             └─────────────┘            └─────────────┘            └─────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               ▼
                                        ┌─────────────┐
                                        │   Hybrid    │
                                        │   Search    │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  LLM / QA   │
                                        │  (Flan-T5)  │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Answer    │
                                        │ + Citations │
                                        └─────────────┘
```

---

## 🧠 Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Embeddings | `all-MiniLM-L6-v2` | 80MB | Convert text to vectors |
| LLM | `flan-t5-small` | 80MB | Generate natural language answers |
| OCR | Tesseract | - | Extract text from images |

**Note:** All models run locally — no API keys or cloud services required!

---

## ⚠️ Troubleshooting

### Memory Error: "Paging file is too small"

This occurs when your system doesn't have enough RAM to load the LLM model.

**Solutions:**
1. The app automatically falls back to **SimpleQA** mode (works without LLM)
2. Use the smaller model (already default): `google/flan-t5-small`
3. Increase Windows virtual memory (see `MEMORY_OPTIMIZATION.md`)

### Deprecation Warnings

If you see LangChain deprecation warnings, update your packages:
```bash
pip install -U langchain-huggingface langchain-community
```

### Tokenizer Loading Error

If the tokenizer fails to load, delete the cache and retry:
```bash
# Windows
rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--google--flan-t5-small

# Linux/Mac
rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-small
```

Then run the app again — it will re-download the model.

---

## 📦 Dependencies

```
streamlit
python-dotenv
pymupdf
pytesseract
Pillow
langchain
langchain-community
langchain-huggingface
langchain-core
faiss-cpu
sentence-transformers
transformers
torch
huggingface-hub
pandas
numpy
rank_bm25
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

**Ask about the document:**
- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the economic indicators mentioned?"
- "Explain the recommendations in section 3"

**The app will:**
1. Search for relevant chunks using hybrid BM25 + Vector search
2. Generate an answer using the LLM (or provide excerpts with SimpleQA)
3. Show citations with source pages and relevance scores

---

## 📄 License

MIT License - Feel free to use and modify for your projects!

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review `MEMORY_OPTIMIZATION.md` for memory issues
3. Open an issue with your error message and system details

---

**Built with ❤️ using Streamlit, LangChain, and HuggingFace**

