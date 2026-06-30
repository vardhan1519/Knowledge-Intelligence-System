from app.models.vector_store import VectorStore
from app.services.storage_service import S3StorageService
from app.services.LLMAdvanced import AutomatedLLMService
from app.config import Config
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import tempfile
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3StorageService()
llm_service = AutomatedLLMService(vector_store)

@app.route('/')
def index():
    return render_template('index.html')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_documents(file):
    """Process the uploaded document and split it into chunks and return chunks."""
    # Save the uploaded file to a temporary location
    temp_dir=tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        file.save(temp_file_path)
        logger.info(f"Saved uploaded file to {temp_file_path}")

        # Upload file to AWS S3 bucket
        try:
            logger.info(f"Uploading '{file.filename}' to S3...")
            s3_success = storage_service.upload_file(temp_file_path, file.filename)
            if s3_success:
                logger.info(f"Successfully uploaded '{file.filename}' to S3.")
            else:
                logger.error(f"Failed to upload '{file.filename}' to S3.")
        except Exception as s3_err:
            logger.error(f"S3 Upload failed with error: {str(s3_err)}")

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        chunks = process_documents(file)
        texts = [chunk.page_content for chunk in chunks]
        if texts:
            vector_store.add_document(texts)
            return jsonify({"message": f"Successfully processed '{file.filename}' and added {len(texts)} chunks to knowledge base."})
        else:
            return jsonify({"error": "No readable content found in file."}), 400
    except Exception as e:
        logger.exception("Failed to process uploaded file")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json() or {}
    question = data.get('question')
    session_id = data.get('session_id', 'default-session')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        response = llm_service.ask(session_id, question)
        return jsonify({"answer": response})
    except Exception as e:
        logger.exception("Failed to query RAG model")
        return jsonify({"error": f"Query execution failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)