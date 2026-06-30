import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import Config

class VectorStore:
    def __init__(self,path):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=path)

    def add_document(self, documents):
        self.vector_store.add_documents(documents)

    def search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)