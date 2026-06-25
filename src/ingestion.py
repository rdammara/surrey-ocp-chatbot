import os
import sys
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter # <--- Added this
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

try:
    if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Missing API keys in .env file.")

    #SETUP EMBEDDING MODEL
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-001",
        embed_batch_size=40)

    #SETUP PARSER (AI-Based Layout Aware OCR)
    print("Reading PDFs with LlamaParse...")
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"), 
        result_type="markdown"
    )
    
    data_dir = ROOT_DIR / "data"
    if not data_dir.exists() or not any(data_dir.glob("*.pdf")):
        raise FileNotFoundError("No PDF found in the data folder.")

    documents = SimpleDirectoryReader(str(data_dir), file_extractor={".pdf": parser}).load_data()

    #EXPLICIT SEGMENTATION (The Chunking Algorithm)
    #define the "Sliding Window" parameters
    print("Segmenting documents into semantic chunks...")
    text_parser = SentenceSplitter(
        chunk_size=1024,  # Number of tokens per chunk
        chunk_overlap=50  # Overlap to maintain context between chunks
    )
    
    #transform raw documents into "Nodes" (the actual chunks)
    nodes = text_parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} semantic nodes.")

    #setup chromadB (ANN / HNSW Indexing)
    db_path = str(ROOT_DIR / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection("surrey_ocp")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. CREATE INDEX FROM NODES (Transformation & Loading)
    # We pass 'nodes' directly instead of 'documents'
    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context,
        show_progress=True
    )

    print(f"Database saved locally at: {db_path} successfully!")

except Exception as e:
    print(f"\n ERROR: {e}\n")
    sys.exit(1)