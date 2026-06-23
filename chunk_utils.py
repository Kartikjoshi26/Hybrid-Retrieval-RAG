from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import streamlit as st
from langsmith import traceable

@st.cache_data
@traceable(name="get clean chunks")
def get_clean_chunks(text,chunk_size=1200,chunk_overlap=240):

    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,     
                chunk_overlap=chunk_overlap,   
                separators=["\n\n", "\n", " ", ""],  
                keep_separator=False
    )
                
    chunks = text_splitter.split_text(text)
    # print(chunks)
    clean_chunks = []
    for doc in chunks:
        # clean_content = re.sub(r'\n+', ' ', doc)
        clean_content = re.sub(r'(?<!\n)\n(?!\n)', ' ', doc)  # Replace single \n with ' '
        clean_content = re.sub(r'\n{2,}', '\n\n', clean_content)  # Normalize multiple \n to double \n
        clean_chunks.append(clean_content)
        # # print("clean_chunks: ")
        # # print(clean_chunks[:6])

    return clean_chunks