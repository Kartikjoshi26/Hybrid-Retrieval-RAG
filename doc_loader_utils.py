import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langsmith import traceable


@traceable(name="build loader")
def get_documents(tmp_path):
    loader = PyPDFLoader(tmp_path)
    return loader.load()