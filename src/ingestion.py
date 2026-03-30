import os
import sys
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# using try-catch for error catching
try:
    if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Missing API keys in .env file.")

    
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name = "models/gemini-embedding-001",
        embed_batch_size = 40)

    #Setup parser & PDF read
    print("Reading PDFs with LlamaParse... This may take a few minutes.")
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown"
        )
    data_dir = ROOT_DIR / "data"
    
    if not data_dir.exists() or not any(data_dir.glob("*.pdf")):
        raise FileNotFoundError("No PDF found in the data folder.")

    documents = SimpleDirectoryReader(str(data_dir), file_extractor={".pdf": parser}).load_data()
    print(f"Parsed {len(documents)} document chunks. Building vector database...")

    #Setup ChromaDB database
    db_path = str(ROOT_DIR / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("surrey_ocp")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #Create Index
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )

    print(f"Database saved locally at: {db_path} sucessfully!")

except Exception as e:
    print(f"\n ERROR: {e}\n")
    sys.exit(1)