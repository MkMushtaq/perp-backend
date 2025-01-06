from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv

app = Flask(__name__)

app = Flask(__name__)
CORS(app) 

# Get API key from .env file
# load env variables
load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")

# Configure paths
TRANSCRIPTS_DIR = "transcripts"
STORAGE_DIR = "storage"
Path(STORAGE_DIR).mkdir(exist_ok=True)

# Configure LlamaIndex settings
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

def initialize_index():
    """Initialize or load the vector store index"""
    try:
        # Try to load existing index
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Loaded existing index")
    except:
        print("Creating new index...")
        # Load documents
        documents = SimpleDirectoryReader(TRANSCRIPTS_DIR).load_data()
        
        # Create and persist index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Created and persisted new index")
    
    return index

# Initialize index on startup
INDEX = initialize_index()

@app.route("/query", methods=["POST"])
def query_transcripts():
    """Endpoint to query the podcast transcripts"""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        
        # Query the index
        query_engine = INDEX.as_query_engine(
            similarity_top_k=3,
            streaming=False
        )
        response = query_engine.query(query)
        
        return jsonify({
            "response": str(response),
            "sources": [node.metadata.get("file_name", "Unknown") 
                       for node in response.source_nodes]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)