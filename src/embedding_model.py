from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


def get_embedding(text: str) -> list:
    model = load_model()
    if not text or not text.strip():
        return [0.0] * 384
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def get_batch_embeddings(texts: list) -> list:
    model = load_model()
    if not texts:
        return []
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()


def compute_similarity(vec1: list, vec2: list) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b))
