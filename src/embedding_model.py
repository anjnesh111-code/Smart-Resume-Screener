from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union



MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[EmbeddingModel] Loading '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)
print("[EmbeddingModel] Model ready.")


def get_embedding(text: str) -> list:

    if not text or not text.strip():
        return [0.0] * 384

    
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

   
    return embedding.tolist()


def get_batch_embeddings(texts: list) -> list:

    if not texts:
        return []

   
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,  # Shows a tqdm progress bar during encoding
    )

   
    return embeddings.tolist()


def compute_similarity(vec1: list, vec2: list) -> float:
    
    a = np.array(vec1)
    b = np.array(vec2)

    return float(np.dot(a, b))


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):

    return model.encode(text).tolist()
   