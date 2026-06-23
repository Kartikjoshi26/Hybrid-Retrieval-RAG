from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever,ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langsmith import traceable


@traceable(name="MMR retriever")
def get_retriever(vector_store,k=5):

    return vector_store.as_retriever(
                search_type="mmr",  
                search_kwargs={
                    "k": k             
                }
            )


def CrossEncoderCompressor(ensemble_retriever):

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(model=model, top_n=4)

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    return compression_retriever


@traceable(name="BM25 retriever")
def get_BM25retriever(vector_store,k=5):
    db_data = vector_store.get()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(db_data.get("documents", []), db_data.get("metadatas", []))
    ]
    return BM25Retriever.from_documents(docs,k=k)



store = InMemoryStore()

@traceable(name="parent child mmr retriever")
def parent_child_mmr_retriever(vector_store,child_splitter,parent_splitter,k=5):

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": 20} 
    )

    return retriever



@traceable(name="Build Hybrid retriever")
def hybrid_retriever(vector_store,raw_documents, k=5):

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)  

    parent_mmr_retriever = parent_child_mmr_retriever(vector_store,child_splitter,parent_splitter,k=5)
    parent_mmr_retriever.add_documents(raw_documents)

    bm25_retriever = get_BM25retriever(vector_store,k=k)
    
    mmr_retriever = get_retriever(vector_store,k=k)

    ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,mmr_retriever, parent_mmr_retriever],
    weights=[0.25,0.25 ,0.5] 
    )

    return ensemble_retriever
    



@traceable(name="Compression retriever")
def hybrid_retriever_with_compression(vector_store,raw_documents, k=5):

    ensemble_retriever = hybrid_retriever(vector_store,raw_documents, k)

    retriever = CrossEncoderCompressor(ensemble_retriever)

    return retriever



