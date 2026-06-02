import os
import sys
import time
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# RAGAS Imports
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# Patch async loops to prevent Windows/VS Code crashing
nest_asyncio.apply()

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

def run_benchmark():
    print("Initializing 4-Way Multi-Model RAG Benchmarking Engine...")

    # 1. VALIDATE KEYS
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
        print("ERROR: Missing GOOGLE_API_KEY or GROQ_API_KEY in .env file.")
        sys.exit(1)

    # 2. LOAD GOLDEN DATASET
    dataset_path = ROOT_DIR / "golden_dataset.json"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Please create it first.")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)
    
    # LlamaIndex wraps the data in an "examples" key, so we extract just the list
    if isinstance(raw_data, dict) and "examples" in raw_data:
        golden_data = raw_data["examples"]
    else:
        golden_data = raw_data # Fallback just in case you use a manual list
        
    print(f"Loaded {len(golden_data)} test questions from Golden Dataset.")

    # 3. CONNECT TO LOCAL CHROMADB
    print("Connecting to local ChromaDB vector store...")
    db_path = str(ROOT_DIR / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection("surrey_ocp")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Fixed Embedding Model matching your ingestion pipeline
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

   # 4. DEFINE THE BENCHMARKING QUAD
    models_to_test = {
        "Gemini_2.5_Flash_Live": GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.0),
        "Gemini_3_Flash_Preview": GoogleGenAI(model="models/gemini-3-flash-preview", temperature=0.0),
        "Llama_3.3_70B_Groq": Groq(model="llama-3.3-70b-versatile", temperature=0.0), # <-- The new active model!
        "Gemini_2.5_Pro_Ceiling": GoogleGenAI(model="models/gemini-2.5-pro", temperature=0.0)
    }

    # Master list to collect row-by-row data points
    master_results = []

    # 5. EXECUTE THE EXPERIMENT LOGIC
    for model_name, llm_instance in models_to_test.items():
        print(f"\nEvaluating Model: {model_name}...")
        Settings.llm = llm_instance
        query_engine = index.as_query_engine(similarity_top_k=2)

        for idx, item in enumerate(golden_data):
            query = item["query"]
            ground_truth = item["reference_answer"]
            
            print(f"  └─ Processing Question {idx + 1}/{len(golden_data)}...", end="", flush=True)
            
            start_time = time.time()
            success = 1
            generated_answer = ""
            retrieved_contexts = []

            try:
                # Fire the RAG query
                response = query_engine.query(query)
                generated_answer = response.response
                retrieved_contexts = [node.node.text for node in response.source_nodes]
            except Exception as e:
                print(f" [FAILED: {e}]")
                success = 0
                generated_answer = "ERROR_TIMEOUT_OR_CRASH"
                retrieved_contexts = ["ERROR"]
            
            latency = time.time() - start_time
            if success:
                print(f" [SUCCESS - {latency:.2f}s]")

            # Append the data to our collection list
            master_results.append({
                "model": model_name,
                "question": query,
                "answer": generated_answer,
                "contexts": retrieved_contexts,
                "ground_truth": ground_truth,
                "latency_seconds": round(latency, 3),
                "success_rate": success
            })
            
            # Rate limit buffer to stay safely within free tiers (especially Groq and Preview endpoints)
            time.sleep(2)

    # 6. RUN RAGAS SEMANTIC EVALUATION USING THE JUDGE MODEL
    print("\nInitializing RAGAS Semantic Judge (Gemini 2.5 Pro)...")
    judge_llm = GoogleGenAI(model="models/gemini-2.5-pro", temperature=0.0)
    
    ragas_judge_llm = LlamaIndexLLMWrapper(judge_llm)
    ragas_judge_emb = LlamaIndexEmbeddingsWrapper(Settings.embed_model)

    # Convert our master list into a Pandas DataFrame, then a HuggingFace Dataset for RAGAS
    df_all = pd.DataFrame(master_results)
    
    # We only send successful queries to RAGAS to avoid calculation mathematical crashes
    df_success = df_all[df_all["success_rate"] == 1].copy()
    
    if not df_success.empty:
        print(f"Computing RAGAS scores for the {len(df_success)} successful outputs (This will take a moment)...")
        eval_dataset = Dataset.from_pandas(df_success)

        # Compute semantic scores
        ragas_output = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=ragas_judge_llm,
            embeddings=ragas_judge_emb
        )
        
        # RAGAS strips out custom columns, but preserves row order. 
        # We extract just the scores and manually attach them back to our success dataframe.
        df_scores = ragas_output.to_pandas()
        
        # Safely map metrics if they exist
        for metric in ['faithfulness', 'answer_relevancy', 'context_recall']:
            if metric in df_scores.columns:
                df_success[metric] = df_scores[metric].values
            else:
                df_success[metric] = None
        
        # Re-combine the scored rows back into the main dataset (to include the FAILED rows)
        final_df = pd.merge(
            df_all, 
            df_success[['model', 'question', 'faithfulness', 'answer_relevancy', 'context_recall']], 
            on=['model', 'question'], 
            how='left'
        )
    else:
        print("⚠ No successful query responses to score with RAGAS.")
        final_df = df_all

    # 7. EXPORT COMPREHENSIVE STATISTICAL DATA
    output_csv = ROOT_DIR / "ragas_comprehensive_benchmark.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"\nEXPERIMENT COMPLETE! Statistical matrix exported to: {output_csv}")
    
    # Print out summary statistics grouped by model for immediate analysis
    print("\nAggregated Performance Metrics Summary:")
    summary = final_df.groupby("model").agg({
        "latency_seconds": "mean",
        "success_rate": "mean",
        "faithfulness": "mean",
        "answer_relevancy": "mean",
        "context_recall": "mean"
    }).round(3)
    print(summary)

if __name__ == "__main__":
    run_benchmark()