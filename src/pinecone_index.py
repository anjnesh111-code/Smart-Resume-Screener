

import os
from pinecone import Pinecone, ServerlessSpec
from typing import Optional



PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY", "pcsk_2HMUbo_GjKx4mB746Jx5MjwxC8bHdfiMf2jNtBpQSuWRd6fwp18155WecnUdqKhKFPnUAC")
INDEX_NAME        = "resume-index"
EMBEDDING_DIM     = 384     
METRIC            = "cosine" 


def get_pinecone_client() -> Pinecone:

    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def create_index_if_not_exists(pc: Pinecone) -> None:

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"[Pinecone] Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[Pinecone] Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"[Pinecone] Index '{INDEX_NAME}' already exists. Skipping creation.")


def get_index(pc: Pinecone):

    return pc.Index(INDEX_NAME)


def upsert_resume(index, resume_id: str, embedding: list, metadata: dict) -> None:

    index.upsert(
        vectors=[
            {
                "id":       resume_id,
                "values":   embedding,
                "metadata": metadata,
            }
        ]
    )


def upsert_batch(index, vectors: list) -> None:

    
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"[Pinecone] Upserted batch {i // batch_size + 1} "
              f"({len(batch)} vectors)")


def query_similar_resumes(index, query_embedding: list, top_k: int = 5) -> list:
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,  # Also return the metadata we stored
    )

    matches = []
    for match in results.matches:
        matches.append({
            "id":       match.id,
            "score":    round(match.score, 4),
            "metadata": match.metadata,
        })

    return matches


def get_index_stats(index) -> dict:
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": EMBEDDING_DIM,
    }
