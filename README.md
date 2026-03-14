# AI Resume Screener with Semantic Search

A production-style ML system that matches job descriptions to resumes
using Sentence Transformer embeddings and Pinecone vector search.

## System Architecture
```
Resume Dataset → Preprocessing → Embeddings → Pinecone → Ranking → Streamlit
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Sentence Transformers | Semantic embeddings (all-MiniLM-L6-v2) |
| Pinecone | Vector database and ANN search |
| spaCy | Text preprocessing and NER |
| Streamlit | Recruiter web interface |
| Pandas / NumPy | Data handling |

## Dataset

- Source: Kaggle (snehaanbhawal/resume-dataset)
- 2484 resumes across 24 profession categories

## Evaluation Results

| Category | Precision@5 | Top Score |
|---|---|---|
| HR | 100% | 70% |
| Accountant | 100% | 77% |
| Teacher | 80% | 62% |
| Banking | 40% | 64% |
| IT | 40% | 50% |

## Project Structure
```
smart-resume-screener/
├── src/
│   ├── preprocessing.py    # Text cleaning and lemmatization
│   ├── embedding_model.py  # Sentence Transformer wrapper
│   ├── pinecone_index.py   # Pinecone operations
│   ├── ranking_engine.py   # Indexing and ranking pipeline
│   ├── app.py              # Streamlit web dashboard
│   └── requirements.txt    # Dependencies
└── final_evaluation.png    # Evaluation charts
```

## Setup
```bash
pip install -r src/requirements.txt
python -m spacy download en_core_web_sm
```

Set your Pinecone API key:
```python
import os
os.environ["PINECONE_API_KEY"] = "your_key_here"
```

Index resumes (one-time):
```python
from src.ranking_engine import index_all_resumes
index_all_resumes()
```

Launch the app:
```bash
streamlit run src/app.py
```

## Key Concepts Demonstrated

- Semantic search using dense vector embeddings
- Approximate Nearest Neighbor search with HNSW indexing
- Transformer-based NLP (Sentence Transformers)
- Vector databases vs traditional keyword search
- Text preprocessing (tokenization, lemmatization, stopword removal)
- Information retrieval evaluation using Precision@K

## Future Improvements

- Resume PDF upload via pdfplumber
- Skill-overlap scoring on top of semantic similarity
- LLM-generated candidate fit explanation using RAG
- Fine-tuned domain-specific embedding model
