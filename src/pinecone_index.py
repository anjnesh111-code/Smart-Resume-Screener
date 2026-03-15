import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME    = "resume-index"
EMBEDDING_DIM = 384
METRIC        = "cosine"


def _get_api_key() -> str:
    try:
        return st.secrets["PINECONE_API_KEY"]
    except Exception:
        key = os.getenv("PINECONE_API_KEY", "")
        if not key:
            raise ValueError(
                "PINECONE_API_KEY not found. "
                "Add it to Streamlit secrets or set it as an environment variable."
            )
        return key


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=_get_api_key())


def create_index_if_not_exists(pc: Pinecone) -> None:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_index(pc: Pinecone):
    return pc.Index(INDEX_NAME)


def upsert_resume(index, resume_id: str, embedding: list, metadata: dict) -> None:
    index.upsert(vectors=[{"id": resume_id, "values": embedding, "metadata": metadata}])


def upsert_batch(index, vectors: list) -> None:
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i: i + batch_size])


def query_similar_resumes(index, query_embedding: list, top_k: int = 5) -> list:
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [
        {"id": m.id, "score": round(m.score, 4), "metadata": m.metadata}
        for m in results.matches
    ]


def get_index_stats(index) -> dict:
    stats = index.describe_index_stats()
    return {"total_vector_count": stats.total_vector_count, "dimension": EMBEDDING_DIM}

def delete_all_vectors(index) -> None:
    index.delete(delete_all=True)
