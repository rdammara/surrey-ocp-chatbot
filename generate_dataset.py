import os
import sys
import random
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio

# LlamaIndex Imports
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

# Apply async patch to prevent VS Code/Windows event loop crashes
nest_asyncio.apply()

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

def generate_test_data():
    print("Initializing Golden Dataset Generator...")

    # --- API KEY VALIDATION ---
    try:
        if not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
            raise ValueError("Missing API keys in .env file. Please check LLAMA_CLOUD_API_KEY, GOOGLE_API_KEY, and GROQ_API_KEY.")
    except Exception as e:
        print(f"\n ERROR: {e}\n")
        sys.exit(1)

    # 1. SETUP THE "EXAM WRITER" MODEL
    # We use Gemini 2.5 Flash to generate the test because it is fast and cheap/free
    Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.3)
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

    # 2. LOAD & CHUNK DATA (Mirroring your ingestion.py)
    print("Reading OCP PDFs...")
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown")
    data_dir = ROOT_DIR / "data"
    
    if not data_dir.exists() or not any(data_dir.glob("*.pdf")):
        raise FileNotFoundError("No PDF found in the data folder.")

    documents = SimpleDirectoryReader(str(data_dir), file_extractor={".pdf": parser}).load_data()

    # Explicit Segmentation (must match your database)
    text_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    nodes = text_parser.get_nodes_from_documents(documents)
    print(f"Total semantic nodes available: {len(nodes)}")

    # 3. SCIENTIFIC SAMPLING
    # We randomly select 30 chunks to generate questions from. 
    # random.seed(42) ensures you get the exact same 10 questions if you run this again
    random.seed(42) 
    sample_size = min(10, len(nodes))
    sampled_nodes = random.sample(nodes, sample_size)
    print(f"Sampled {sample_size} nodes for the exam.")

    # 4. GENERATE THE DATASET
    print("Writing the exam questions and ground truths (This will take 1-3 minutes)...")
    dataset_generator = RagDatasetGenerator(
        nodes=sampled_nodes,
        llm=Settings.llm,
        num_questions_per_chunk=1, # Generate 1 question per selected paragraph
        show_progress=True
    )

    # Execute the generation
    rag_dataset = dataset_generator.generate_dataset_from_nodes()

    # 5. SAVE THE ARTIFACT
    output_path = ROOT_DIR / "golden_dataset.json"
    rag_dataset.save_json(str(output_path))
    print(f"\nSUCCESS! Golden Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_test_data()