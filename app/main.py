from models.vector_store import VectorStore
from services.storage_service import S3StorageService
from services.LLMAdvanced import AutomatedLLMService
from config import Config
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import tempfile

vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3StorageService()
llm_service = AutomatedLLMService(vector_store)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_and_store_document(file):

    # Save the uploaded file to a temporary location
    temp_dir=tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        file.save(temp_file_path)
        logger.info(f"Saved uploaded file to {temp_file_path}")

        # Determine the loader based on file type
        if file.filename.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        elif file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif file.filename.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        else:
            raise ValueError("Unsupported file type")

        # Load and split the document into chunks
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Deleted temporary file {temp_file_path}")
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
            logger.info(f"Deleted temporary directory {temp_dir}")