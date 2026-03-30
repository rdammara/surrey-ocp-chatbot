import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# 1. Setup paths
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

def get_query_engine():
    """Loads the database and connects it to Gemini 3 Flash."""
    
    # 2. Configure Models (Must match ingestion.py exactly!)
    Settings.llm = GoogleGenAI(
        model="models/gemini-3-flash-preview", 
        api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries = 3
    )
    # Note: Use 'gemini-embedding-001' to match your successful ingestion
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

    # 3. Connect to the existing ChromaDB folder
    db_path = str(ROOT_DIR / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Retrieve the collection we created earlier
    chroma_collection = chroma_client.get_collection("surrey_ocp")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # 5. Return the engine 
    # 'similarity_top_k=5' means it pulls the 5 most relevant 
    # paragraphs from your PDFs to answer each question.
    return index.as_query_engine(similarity_top_k = 2, streaming = True)