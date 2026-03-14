import pdfplumber
from embedding_model import get_embedding
from pinecone_index import index

def extract_text(file):

    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


def add_resume(file, resume_id):

    text = extract_text(file)

    embedding = get_embedding(text)

    index.upsert([
        (resume_id, embedding)
    ])

    return text
