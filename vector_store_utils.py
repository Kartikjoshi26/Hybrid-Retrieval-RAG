from langchain_community.vectorstores import Chroma
from langsmith import traceable

@traceable(name="build vector store")
def get_vector_store(embedding_model,vectorstore_documents = None):

    if vectorstore_documents:
        return Chroma.from_documents(
            embedding=embedding_model,
            documents=vectorstore_documents
        )

    vector_store = Chroma(embedding_function=embedding_model)

    return vector_store

