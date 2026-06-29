import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import Config

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=Config.VECTOR_DB_PATH)

    def add_document(self, documents):
        self.vector_store.add_texts(documents)

    def search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)