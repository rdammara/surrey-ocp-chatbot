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
    
    #Configure Models (Must match ingestion.py exactly!)
    Settings.llm = GoogleGenAI(
        model="models/gemini-2.5-flash", 
        api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries = 3,
        request_options = {"timeout": 15.0},
        system_instruction=(
            "You are an expert urban planning assistant for the City of Surrey. "
            "Your primary goal is to help residents understand the 2050 Official Community Plan (OCP). "
            "Base all your answers strictly on the retrieved context. "
            "\n\nCRITICAL RULES:"
            "\n1. LIABILITY DISCLAIMER: If the user asks about land use, building heights, or property rules, "
            "you must append this exact disclaimer at the bottom of your response: "
            "'*Disclaimer: The OCP is a high-level guiding document. For specific zoning regulations, "
            "legal allowances, or building permits for your property, please consult official City of Surrey staff.*' "
            "\n2. SPECIFIC ADDRESSES: The OCP does not dictate rules for individual houses. If a user asks about "
            "a specific street address, you must kindly tell them you cannot look up individual parcels and direct "
            "them to use the official City of Surrey COSMOS mapping system."
        )
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