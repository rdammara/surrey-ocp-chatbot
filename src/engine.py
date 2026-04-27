import os
import chromadb
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

#File directory setup paths
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

#grab the key on local OR cloud (in this case Streamlit)
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_query_engine():
    """Loads database...."""
    
    # 1. Configure Models (Using the safely captured GOOGLE_API_KEY variable)
    Settings.llm = GoogleGenAI(
        model="models/gemini-2.5-flash", 
        api_key=GOOGLE_API_KEY,  # <-- FIX 1: Using your safe variable
        max_retries = 3,
        request_options = {"timeout": 15.0},
        system_instruction=(
            "You are an expert urban planning assistant for the City of Surrey. "
            "Your primary goal is to help residents understand the 2050 Official Community Plan (OCP). "
            "Base all your answers strictly on the retrieved context."
            # Removed the duplicate rules because app.py handles them perfectly now!
        )
    )
    
    # FIX 2: Explicitly pass the key to the embedding model
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=GOOGLE_API_KEY 
    )

    # FIX 3: Turn off ChromaDB telemetry to stop the local PC freezing
    db_path = str(ROOT_DIR / "chroma_db")
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    
    # Retrieve the collection we created earlier
    chroma_collection = chroma_client.get_collection("surrey_ocp")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Return the engine 
    return index.as_query_engine(similarity_top_k=2, streaming=True)