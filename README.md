# Surrey AI-Powered Official Community Plan (OCP) Chatbot Assistant

**Focus:** Enhancing Civic Engagement through implementation of Large Language Model (LLM) and Retrieval-Augmented Generation (RAG)



## Project Overview
This project addresses the complexity of urban planning documentation by providing an accessible, AI-driven interface for the **City of Surrey’s 2050 Official Community Plan (Draft 2026)**. 

Using the **Gemini 3 Flash** LLM and a **RAG (Retrieval-Augmented Generation)** pipeline, this web app allows residents to ask natural language questions and receive accurate, cited answers directly from the 500+ page OCP and its associated engagement reports (Phases 1-4).

## Tech Stack
* **Language Model:** Gemini 3 Flash (via Google AI Studio)
* **Framework:** LlamaIndex (Data Orchestration & RAG)
* **Vector Database:** ChromaDB (Local Vector Storage)
* **Frontend:** Streamlit (Web Interface)
* **Deployment:** GitHub & Streamlit Cloud

## Key Features (MVP)
- **Policy Pinpointing:** Ask about specific neighborhoods (e.g., Fleetwood, Cloverdale) and get direct policy summaries.
- **Source Citations:** Every answer includes the specific PDF document and page number to prevent hallucinations.
- **Engagement Insights:** TModel awareness of resident concerns from the Phase 1-4 Engagement Surveys, allowing it to prioritize confusing topics.

## Repository Structure
```text
├── data/               # Official Surrey OCP PDF documents
├── src/                # Backend RAG logic and PDF processing
├── app.py              # Main Streamlit web application
├── requirements.txt    # Python dependencies
└── .gitignore          # Security: Prevents API keys from being leaked
