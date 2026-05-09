# Medical RAG Chatbot 🏥💊

A Retrieval-Augmented Generation (RAG) chatbot for medical Q&A, built on MedQuAD and DrugBank datasets.

## Architecture

```
User Query
    │
    ▼
Intent Detection ──► Drug interaction? Symptom? General?
    │
    ▼
Retriever (Hybrid: FAISS dense + TF-IDF sparse)
    │                  │
  MedQuAD          DrugBank
  (Q&A pairs)   (Drug interactions)
    │
    ▼
Context Injection → Prompt Engineering
    │
    ▼
LLM API (GPT-4o / Gemini)
    │
    ▼
Streamlit Chat UI  ◄── Sources + Severity badges
```

## Project Structure

```
medical_rag_chatbot/
│
├── app.py                        # Streamlit UI entry point
├── build_index.py                # One-time index builder
├── requirements.txt
├── .env.example                  # Copy to .env and fill API keys
│
├── data/
│   └── preprocessor.py          # MedQuAD + DrugBank parsing & chunking
│
├── embeddings/
│   └── vector_store.py          # Sentence-transformer + FAISS index
│                                #   (includes HybridRetriever)
│
├── retrieval/
│   └── tfidf_retriever.py       # Sparse TF-IDF baseline
│
└── llm/
    ├── llm_client.py            # OpenAI + Gemini unified client
    ├── prompt_engine.py         # Context injection + prompt engineering
    └── rag_pipeline.py          # End-to-end pipeline (wires everything)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or GOOGLE_API_KEY
```

### 3. (Optional) Add real datasets

**MedQuAD** — download from https://github.com/abachaa/MedQuAD
```bash
mkdir -p data/medquad
# Place XML files in data/medquad/
# Set MEDQUAD_PATH=./data/medquad in .env
```

**DrugBank** — register at https://go.drugbank.com/releases/latest
```bash
# Download the CSV export, place it as data/drugbank.csv
# Set DRUGBANK_CSV_PATH=./data/drugbank.csv in .env
```

If you skip this step, the app uses built-in sample data (8 conditions + 5 drugs) which is enough for testing.

### 4. Build the vector index

```bash
python build_index.py
# With tests:
python build_index.py --test
# Force rebuild:
python build_index.py --rebuild
```

### 5. Run the chatbot

```bash
streamlit run app.py
```

## Features

| Feature | Details |
|---|---|
| Dual data sources | MedQuAD Q&A pairs + DrugBank drug profiles |
| Hybrid retrieval | FAISS dense (sentence-transformers) + TF-IDF sparse, fused via RRF |
| Intent detection | Automatically detects drug interaction, symptom, or general queries |
| Drug interaction logic | Severity scoring (contraindicated / major / moderate / minor) |
| Multi-LLM support | GPT-4o, GPT-4o-mini, Gemini 1.5 Pro/Flash, or Mock mode |
| Streaming responses | Real-time token streaming via Streamlit |
| Source citations | Every answer shows which chunks were retrieved |
| Multi-turn memory | Last 3 conversation turns included as context |

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GOOGLE_API_KEY` | — | Google AI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Model to use |
| `VECTOR_STORE_PATH` | `./embeddings/faiss.index` | FAISS index path |
| `CHUNKS_PATH` | `./embeddings/chunks.json` | Chunks JSON path |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SCORE_THRESHOLD` | `0.35` | Minimum similarity score |
| `MEDQUAD_PATH` | — | Path to MedQuAD XML directory |
| `DRUGBANK_CSV_PATH` | — | Path to DrugBank CSV |

## Embedding Models

Edit `vector_store.py` `DEFAULT_MODEL` to change:

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | ⚡ Fast | Good | Default, 384-dim |
| `all-mpnet-base-v2` | Medium | Better | 768-dim |
| `pritamdeka/S-PubMedBert-MS-MARCO` | Slow | Best | Biomedical domain |

## No API Key? Use Mock Mode

Set `LLM_MODEL=mock` in `.env` or select "mock" in the sidebar. The app will work fully for testing retrieval, UI, and sources — just without real LLM responses.

## Sample Questions to Try

- "Can I take ibuprofen with warfarin?"
- "What are the symptoms of heart failure?"
- "How does metformin work for type 2 diabetes?"
- "What drugs interact with atorvastatin?"
- "Is it safe to take lisinopril with potassium supplements?"
- "What is COPD and how is it managed?"

## Disclaimer

This tool is for **informational and educational purposes only**. It does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions.
