# RAG Chatbot Project

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain. It allows users to upload PDF documents and ask questions based on their content.

## Features

- **Interactive Chat Interface**: A clean and responsive UI powered by Streamlit.
- **Smart PDF Processing**:
    - **Digital & Scanned Detection**: Automatically detects if a PDF is digital (selectable text) or scanned (images).
    - **OCR Support**: Uses Tesseract OCR to extract text from scanned PDFs.
    - **Table Extraction**: Extracts structured data from tables using Camelot and pdfplumber.
- **Advanced Retrieval**:
    - **Vector Database**: Uses ChromaDB to store and retrieve document chunks.
    - **MMR Search**: Implements Maximal Marginal Relevance (MMR) to ensure diversity in retrieved information.
- **Large Language Model**: Integrated with Groq (specifically Llama-3.3-70b-versatile) for fast and accurate responses.
- **Session History**: Maintains context within a chat session.

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` or update `.env`)

## Getting Started

1. **Activate the Virtual Environment**:
   ```powershell
   .\RAG_project\Scripts\activate
   ```

2. **Install Dependencies** (if not already installed):
   ```powershell
   pip install streamlit langchain langchain-community langchain-groq langchain-huggingface chromadb pytesseract pdfplumber camelot-py[cv] pymupdf pillow python-dotenv
   ```
   *Note: Camelot also requires [Ghostscript](https://camelot-py.readthedocs.io/en/master/user/install.html) to be installed on your system.*

3. **Configure Environment Variables**:
   Ensure your `.env` file contains your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key
   MODEL_NAME=llama-3.3-70b-versatile
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

4. **Run the Application**:
   ```powershell
   streamlit run RAG.py
   ```

## Usage

1. Upload a PDF file using the sidebar.
2. Click **Submit** to process the file.
3. Once processed, type your questions in the chat box at the bottom.
