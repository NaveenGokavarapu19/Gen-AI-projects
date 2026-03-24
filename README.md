# RAG Chatbot — AWS Bedrock + LangChain + FAISS

A production-style Retrieval Augmented Generation (RAG) chatbot 
that answers questions from any uploaded PDF document.

## Architecture
PDF Upload → Chunk → Embed (AWS Bedrock Titan) → FAISS
Question → Embed → Search FAISS → Retrieved Chunks → 
Amazon Nova Lite → Grounded Answer

## Tech Stack
- **LLM**: Amazon Nova Lite (via AWS Bedrock)
- **Embeddings**: Amazon Titan Embed V1 (via AWS Bedrock)
- **Vector Store**: FAISS (local)
- **Framework**: LangChain
- **UI**: Streamlit
- **Language**: Python 3.11

## How It Works
1. Upload any PDF through the sidebar
2. Document is chunked into 1000-character overlapping segments
3. Each chunk is embedded into a 1536-dimensional vector using Titan
4. Vectors stored in FAISS for similarity search
5. Questions are embedded and matched against stored vectors
6. Top 3 relevant chunks sent to Nova Lite with strict context prompt
7. Answer generated only from document content — no hallucination

## Project Structure
rag-chatbot/
  ├── app.py       # Streamlit frontend
  ├── main.py      # Backend logic
  ├── data/        # Sample PDFs
  └── README.md

## Setup
pip install -r requirements.txt
streamlit run app.py

## Key Design Decisions
- chunk_overlap=200 prevents information loss at boundaries
- k=3 retrieval balances context richness vs noise
- Explicit prompt instruction prevents hallucination
- @st.cache_resource prevents re-initializing LLM on every message
- Backend/frontend separation keeps code maintainable