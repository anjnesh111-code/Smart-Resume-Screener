import os
import glob
import pandas as pd
from src.preprocessing import preprocess_resume
from src.embedding_model import get_embedding, get_batch_embeddings
from src.pinecone_index import (
    get_pinecone_client,
    create_index_if_not_exists,
    get_index,
    upsert_batch,
    query_similar_resumes,
    get_index_stats,
)


def load_resumes_from_disk(data_dir: str = "data/resumes") -> pd.DataFrame:
    records = []

    for filepath in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        category = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            records.append({
                "filename": filename,
                "category": category,
                "raw_text": raw_text,
            })
        except Exception as e:
            print(f"[Loader] Warning: Could not read {filepath}: {e}")

    df = pd.DataFrame(records)
    print(f"[Loader] Loaded {len(df)} resumes across {df['category'].nunique()} categories.")
    return df


def index_all_resumes(data_dir: str = "data/resumes") -> None:
    print("\n===== Starting Resume Indexing Pipeline =====")

    df = load_resumes_from_disk(data_dir)
    if df.empty:
        print("[Indexer] No resumes found. Please check your data directory.")
        return

    print("[Indexer] Preprocessing resumes...")
    df["processed_text"] = df["raw_text"].apply(
        lambda text: preprocess_resume(text)["processed"]
    )
    df["skills"] = df["raw_text"].apply(
        lambda text: preprocess_resume(text)["skills"]
    )

    print("[Indexer] Generating embeddings...")
    embeddings = get_batch_embeddings(df["processed_text"].tolist())

    vectors = []
    for i, row in df.iterrows():
        resume_id = f"{row['category']}_{row['filename']}_{i}"
        text_preview = row["raw_text"][:1000]

        vectors.append({
            "id": resume_id,
            "values": embeddings[i],
            "metadata": {
                "category":     row["category"],
                "filename":     row["filename"],
                "text_preview": text_preview,
                "skills":       ", ".join(row["skills"][:20]),
            },
        })

    print("[Indexer] Connecting to Pinecone...")
    pc = get_pinecone_client()
    create_index_if_not_exists(pc)
    index = get_index(pc)

    print("[Indexer] Upserting vectors to Pinecone...")
    upsert_batch(index, vectors)

    stats = get_index_stats(index)
    print(f"\n[Indexer] ✅ Done! Index now contains {stats['total_vector_count']} vectors.")


def rank_candidates(job_description: str, top_k: int = 10) -> pd.DataFrame:
    if not job_description.strip():
        return pd.DataFrame()

    processed_jd = preprocess_resume(job_description)["processed"]

    print("[Ranker] Embedding job description...")
    jd_embedding = get_embedding(processed_jd)

    print(f"[Ranker] Querying Pinecone for top {top_k} matches...")
    pc = get_pinecone_client()
    index = get_index(pc)
    matches = query_similar_resumes(index, jd_embedding, top_k=top_k)

    if not matches:
        print("[Ranker] No matches found.")
        return pd.DataFrame()

    results = []
    for rank, match in enumerate(matches, start=1):
        meta = match.metadata or {}
        category = meta.get("category", "Unknown")
        preview = meta.get("text_preview", "")
        skills = meta.get("skills", "")
            
    df_results = pd.DataFrame(results)
    df_results["match_pct"] = (df_results["score"] * 100).round(1).astype(str) + "%"

    print(f"[Ranker] ✅ Found {len(df_results)} candidates.")
    return df_results
