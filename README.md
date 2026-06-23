# Advanced Hybrid Retrieval RAG Chatbot

An advanced, production-grade Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit** and **LangChain**. This chatbot features modular architecture, parent-child document retrieval, local Cross-Encoder reranking, PII sanitization for enterprise data security, and LangSmith observability.

---

## 🌟 Advanced Features

### 📁 1. Clean Modular Architecture
The codebase has been refactored into focused, modular components:
*   `RAG.py`: The main Streamlit user interface, chat flow, and pipeline orchestrator.
*   `retriever_utils.py`: Contains advanced retrieval logic, including parent-child mapping, BM25, and reranker compression.
*   `security_utils.py`: Handles data anonymization/deanonymization and PII detection.
*   `doc_loader_utils.py` & `pdf_utils.py`: Handle PDF parsing, digital vs. scanned detection, and page-by-page OCR extraction.
*   `vector_store_utils.py`: Manages vector database instantiation and collection handling.
*   `chunk_utils.py`: Houses custom text-splitting configurations.

### 🔎 2. Multi-Way Hybrid Retrieval (Parent-Child Architecture)
Instead of searching over arbitrary text blocks, the retrieval pipeline leverages a **Parent-Child Document Retriever**:
*   **Child Chunks (~300 characters)**: Used for precise, dense semantic vector searches.
*   **Parent Documents (~2000 characters)**: Once the semantic search finds the best child chunk, it resolves back to the full parent paragraph context stored in an `InMemoryStore`.
*   **Ensemble Retrieval**: Combines keyword-based search (**BM25Retriever**) with semantic vector search (**MMR parent retriever**) using weighted scoring (`0.4` BM25 / `0.6` Parent MMR).

### 🚀 3. Local Cross-Encoder Reranking
Reranks the ensembled candidates locally using `cross-encoder/ms-marco-MiniLM-L-6-v2` via `ContextualCompressionRetriever`. This ensures:
*   High-precision document ranking before context is sent to the LLM.
*   No external API reranker costs (runs entirely locally on CPU).
*   Fast, sub-second latency and a small memory footprint (~80 MB).

### 🔒 4. PII Sanitization & Enterprise Privacy
Features a bidirectional PII (Personally Identifiable Information) masking system using **Microsoft Presidio** (`PresidioReversibleAnonymizer`):
*   Detects and redacts sensitive entities: Aadhaar Cards, PIN Codes, Emails, and Credit Cards.
*   Anonymizes the user query and retrieved documents *before* they are sent to the external Groq LLM API.
*   Streams the LLM response safely, and automatically deanonymizes the placeholders back to real values in the Streamlit UI in real-time once the generation is complete.

### 🛡️ 5. Multi-Layered Security Guardrails (Prompt Safety)
Protects the LLM against malicious attacks and unsafe outputs using a defense-in-depth safety pipeline via Hugging Face Serverless Inference:
*   **Prompt Injection & Jailbreak Prevention**: Uses **Llama-Prompt-Guard-2-86M** (or ProtectAI's `deberta-v3-base-prompt-injection-v2`) to intercept user queries and filter out adversarial attempts designed to hijack system instructions.
*   **Content Safety Moderation**: Uses **Llama-Guard-3-8B** to perform real-time checks on user prompts, filtering out harmful domains (e.g., violent crimes, weapons, self-harm, hate speech, etc.).
*   **Indirect Prompt Injection Filtering**: Extends prompt guard classification to retrieved document chunks (`filter_safe_docs`), ensuring malicious payloads embedded in files are removed *before* reaching the LLM context.


### 📄 6. Metadata Propagation & Source Citations
*   Page-level numbers are captured during ingestion (supporting both digital and scanned OCR page parsing).
*   Metadata is preserved across parent-child chunks.
*   The system prompt instructs the model to explicitly cite the source document name and page number for every fact in its answer (e.g. `[Source: document_name.pdf, Page: 4]`).

### 📊 7. LangSmith Observability
Fully decorated with `@traceable` spans to log and organize traces in LangSmith. The ingestion pipeline (`run_ingestion_pipeline`) and query pipeline (`run_rag_pipeline`) are structured hierarchically:
*   Ingestion steps (`build loader`, `get clean chunks`, etc.) are grouped under a single ingestion trace parent.
*   Vector store retrievals and LLM invocations are grouped cleanly under the query trace parent.

---

## 🛠️ Prerequisites

Before getting started, ensure you have the following installed:
*   Python 3.10+
*   [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (installed on your system and mapped in `.env`)
*   *For table extraction, Camelot requires [Ghostscript](https://camelot-py.readthedocs.io/en/master/user/install.html) to be installed on your machine.*

---

## ⚙️ Getting Started

### 1. Activate the Virtual Environment
```powershell
.\RAG_project\Scripts\activate
```

### 2. Install Dependencies
Ensure you install all base libraries and the required Spacy NLP model for PII entity recognition:
```powershell
# Install core and experimental LangChain libraries (compatible with v0.3.x ecosystem)
pip install "langchain-core<0.4.0" "langchain-community<0.4.0" "langchain-experimental<0.4.0" "langchain-huggingface<0.2.0" "langchain-text-splitters<0.4.0" "langchain<0.4.0"

# Install Streamlit, database, OCR, and table extraction tools
pip install streamlit chromadb pytesseract pdfplumber camelot-py[cv] pymupdf pillow python-dotenv sentence-transformers torch transformers rank_bm25

# Install PII analysis packages and download Spacy model
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=llama-3.3-70b-versatile
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Hugging Face Access Token & Safety Models
HF_TOKEN=your_huggingface_access_token_here
CONTENT_MODERATE_MODEL_ID=meta-llama/Llama-Guard-3-8B
PROMPT_GUARD_MODEL_ID=meta-llama/Llama-Prompt-Guard-2-86M

# LangSmith tracing keys
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=RAG-Chatbot
```

### 4. Run the Application
```powershell
streamlit run RAG.py
```

---

## 💡 Usage

1.  Upload a document (PDF) in the sidebar.
2.  Click **Submit** to trigger the modular ingestion pipeline. You will see interactive progress spinners as it extracts text, tables, creates the parent-child mapping, and generates the vector DB.
3.  Once indexed, type your query. The app will sanitize your PII data, retrieve and rerank context, stream the masked response, and output the clean deanonymized result with exact page citations!
