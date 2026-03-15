import os
import glob
import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_resume
from src.embedding_model import get_embedding, get_batch_embeddings
from src.pinecone_index import (
    get_pinecone_client, create_index_if_not_exists,
    get_index, upsert_batch, query_similar_resumes, get_index_stats,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_kaggle_dataset() -> str:
    """Download resume dataset from Kaggle and return path to CSV."""
    try:
        import kaggle
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        import kaggle

    # Set credentials from Streamlit secrets
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

    download_path = os.path.join(BASE_DIR, "data")
    os.makedirs(download_path, exist_ok=True)

    print("[Kaggle] Downloading resume dataset...")
    from kaggle.api.kaggle_api_extended import KaggleApiClient
    import kaggle as kg
    kg.api.authenticate()
    kg.api.dataset_download_files(
        "snehaanbhawal/resume-dataset",
        path=download_path,
        unzip=True,
    )
    print("[Kaggle] Download complete.")

    # Find the CSV
    for name in ["Resume.csv", "resume.csv", "UpdatedResumeDataSet.csv"]:
        path = os.path.join(download_path, name)
        if os.path.exists(path):
            return path

    csvs = glob.glob(os.path.join(download_path, "*.csv"))
    if csvs:
        return csvs[0]

    raise FileNotFoundError("Kaggle download succeeded but no CSV found.")


def load_resumes_from_csv() -> pd.DataFrame:
    """Load resumes from CSV, downloading from Kaggle if needed."""
    csv_path = None

    # Check if already downloaded
    for name in ["Resume.csv", "resume.csv", "UpdatedResumeDataSet.csv", "resume_dataset.csv"]:
        path = os.path.join(BASE_DIR, "data", name)
        if os.path.exists(path):
            csv_path = path
            break

    # Download from Kaggle if not found
    if csv_path is None:
        csv_path = download_kaggle_dataset()

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for col in df.columns:
        if col in ("resume_str", "resume", "resumetext", "text", "content"):
            col_map[col] = "raw_text"
        if col in ("category", "label", "profession", "job_category", "class"):
            col_map[col] = "category"
    df = df.rename(columns=col_map)

    if "raw_text" not in df.columns:
        raise ValueError(f"Could not find resume text column. Columns: {list(df.columns)}")
    if "category" not in df.columns:
        df["category"] = "Unknown"

    df = df[["raw_text", "category"]].dropna(subset=["raw_text"])
    df["filename"] = df.index.astype(str) + ".txt"

    print(f"[Loader] Loaded {len(df)} resumes across {df['category'].nunique()} categories.")
    return df


def load_resumes_from_disk(data_dir: str = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data", "resumes")

    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    if not txt_files:
        return load_resumes_from_csv()

    records = []
    for filepath in txt_files:
        category = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            records.append({"filename": filename, "category": category, "raw_text": raw_text})
        except Exception as e:
            print(f"[Loader] Warning: {filepath}: {e}")

    df = pd.DataFrame(records)
    print(f"[Loader] Loaded {len(df)} resumes across {df['category'].nunique()} categories.")
    return df


def index_all_resumes(data_dir: str = None) -> None:
    df = load_resumes_from_disk(data_dir)
    if df.empty:
        raise ValueError("No resumes found to index.")

    df["processed_text"] = df["raw_text"].apply(lambda t: preprocess_resume(t)["processed"])
    df["skills"] = df["raw_text"].apply(lambda t: preprocess_resume(t)["skills"])
    embeddings = get_batch_embeddings(df["processed_text"].tolist())

    vectors = []
    for i, row in df.iterrows():
        vectors.append({
            "id": f"{row['category']}_{row['filename']}_{i}",
            "values": embeddings[i],
            "metadata": {
                "category":     str(row["category"]),
                "filename":     str(row["filename"]),
                "text_preview": str(row["raw_text"])[:1000],
                "skills":       ", ".join(row["skills"][:20]),
            },
        })

    pc = get_pinecone_client()
    create_index_if_not_exists(pc)
    upsert_batch(get_index(pc), vectors)
    print("[Indexer] Done.")


def rank_candidates(job_description: str, top_k: int = 10) -> pd.DataFrame:
    if not job_description.strip():
        return pd.DataFrame()

    processed_jd = preprocess_resume(job_description)["processed"]
    jd_embedding = get_embedding(processed_jd)

    matches = query_similar_resumes(get_index(get_pinecone_client()), jd_embedding, top_k=top_k)
    if not matches:
        return pd.DataFrame()

    results = []
    for rank, match in enumerate(matches, start=1):
        meta = match.get("metadata", {}) or {}
        results.append({
            "rank":         rank,
            "candidate_id": match.get("id", ""),
            "score":        match.get("score", 0),
            "category":     meta.get("category", "Unknown"),
            "text_preview": meta.get("text_preview", ""),
            "skills":       meta.get("skills", ""),
        })

    df_results = pd.DataFrame(results)
    df_results["match_pct"] = (df_results["score"] * 100).round(1).astype(str) + "%"
    return df_results


