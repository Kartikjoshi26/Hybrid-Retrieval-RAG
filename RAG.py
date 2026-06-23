import streamlit as st
import tempfile
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
import camelot
import pdfplumber

import fitz 
import pytesseract
pytesseract.pytesseract.tesseract_cmd =  os.getenv("TESSERACT_PATH")

from PIL import Image
from io import BytesIO
import random

from langsmith import traceable

from pdf_utils import detect_pdf_type
from prompt_utils import set_System_prompt
from chunk_utils import get_clean_chunks
from retriever_utils import get_retriever, hybrid_retriever_with_compression
from vector_store_utils import get_vector_store
from doc_loader_utils import get_documents
from security_utils import anonymize_text, deanonymize_text, sanitize_docs


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("MODEL_NAME")

text=''

@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("MODEL_NAME")
    )

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

llm = get_llm()
embedding_model = get_embedding_model()


st.header("Welcome to RAG Chatbot")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if 'chatHistory' not in st.session_state:
    st.session_state['chatHistory'] = []


required_type = ['pdf']


@traceable(name="ingestion_pipeline")
def run_ingestion_pipeline(file):
        # text = ""
        raw_documents = []
        table_docs = []
        if file.name.endswith(".pdf"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name

                    pdf_info = detect_pdf_type(tmp_path, sample_pages=5)
        
                    # print(f"PDF Type Detected: {pdf_info['pdf_type']}")

                    if "Scanned" in pdf_info['pdf_type']:

                        doc = fitz.open(tmp_path)
                        ocr_text = ""
                        for page_num, page in enumerate(doc):
                            pix = page.get_pixmap(dpi=200)
                            img = Image.open(BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(img)
                            
                            raw_documents.append(Document(
                                page_content=ocr_text,
                                metadata={"source": file.name, "page": page_num + 1} # 1-indexed page
                             ))
                        # text = ocr_text

                    else:
                        
                        raw_documents = get_documents(tmp_path)
                        # text = "\n".join([doc.page_content for doc in documents])
                        for doc in raw_documents:
                            doc.metadata["source"] = file.name
                            if "page" in doc.metadata:
                                doc.metadata["page"] = doc.metadata["page"] + 1

                st.success("File Uploaded Successfully")

                table_docs = []

                # Camelot extraction
                try:
                    tables = camelot.read_pdf(tmp_path, pages="all")
                    for i, table in enumerate(tables):
                        table_docs.append(Document(
                            page_content=f"Table {i+1}:\n{table.df.to_string(index=False)}",
                            metadata={"source": file.name, "type": "table"}
                        ))
                except:
                    # Fallback to pdfplumber
                    with pdfplumber.open(tmp_path) as pdf:
                        for page_number, page in enumerate(pdf.pages, start=1):
                            tables = page.extract_tables()
                            for i, table in enumerate(tables):
                                text_table = "\n".join([", ".join([cell if cell is not None else "" for cell in row]) for row in table ])
                                table_docs.append(Document(
                                    page_content=f"Page {page_number} Table {i+1}:\n{text_table}",
                                    metadata={"source": file.name, "type": "table"}
                                ))    

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


        # clean_chunks = get_clean_chunks(text)

        # vectorstore_documents = [Document(page_content=c) for c in clean_chunks]

        # vectorstore_documents.extend(table_docs)

        raw_documents.extend(table_docs)

        vector_store = get_vector_store(embedding_model)

        return hybrid_retriever_with_compression(vector_store, raw_documents, k=5)

        


with st.sidebar:
    file = st.file_uploader("Upload your file here",type=required_type)
    if(file):
        # print(file)
        # print(file.name)
        file_ext = file.name.split(".")[-1].lower()

        if st.button("Submit"):
            st.session_state.retriever = run_ingestion_pipeline(file)



user_input = st.chat_input("Ask your question here")



@traceable(run_type="chain", name="RAG Query Pipeline")
def run_rag_pipeline(user_input):


    relevant_docs = st.session_state.retriever.get_relevant_documents(user_input)

    pii_cleaned_docs = sanitize_docs(relevant_docs)

    # clean_relevant_docs = [c.page_content for c in pii_cleaned_docs]
    # print("Relevant docs: ")
    # print(clean_relevant_docs[:5])
    
    clean_user_input = anonymize_text(user_input)
    system_prompt = set_System_prompt(user_input=clean_user_input,clean_relevant_docs=pii_cleaned_docs)

        
    st.session_state['chatHistory'].append(HumanMessage(content=clean_user_input))
        


    try: 
        with st.chat_message('assistant'):

            # result = st.write_stream(
            #             message_chunk.content for message_chunk in llm.stream(
            #                 [system_prompt] + st.session_state['chatHistory']
            #             )
            #         )

            # clean_result = anonymize_text(result)
            # # print("result :",result)
            # st.session_state['chatHistory'].append(AIMessage(content = clean_result))
            # return deanonymize_text(result)
            # # print("Chat History after result :",st.session_state['chatHistory'])


            message_placeholder = st.empty()
            raw_result = ""
            
            # Accumulate the streamed response
            for message_chunk in llm.stream([system_prompt] + st.session_state['chatHistory']):
                raw_result += message_chunk.content
                message_placeholder.markdown(raw_result + "▌") # Show typing indicator
            
            # Deanonymize the final text
            final_result = deanonymize_text(raw_result)
            
            # Overwrite container to display the real (deanonymized) text
            message_placeholder.markdown(final_result)
            
            # Save the anonymized version in chat history to preserve context for the next turn
            clean_result = anonymize_text(raw_result)
            st.session_state['chatHistory'].append(AIMessage(content=clean_result))
            
            return final_result

    except Exception as e:
        st.session_state['chatHistory'].append(AIMessage(content="Sorry, I encountered an error. Please try again."))
        st.error("Error invoking model")
        print(f"Error invoking model: {e}")




for msg in st.session_state['chatHistory']:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)


if user_input:

    with st.chat_message('user'):
        st.text(user_input)


    if st.session_state.retriever is not None:
        result = run_rag_pipeline(user_input)
            
    else:
        st.warning("⚠️ Please upload and submit a file first before asking questions.")

