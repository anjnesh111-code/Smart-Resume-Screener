import re
import string


# Simple keyword-based skill dictionary
SKILL_KEYWORDS = [
    "python", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "pandas", "numpy",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "data analysis", "nlp", "computer vision"
]


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?:\/\/\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text: str) -> list:

    text_lower = text.lower()
    skills_found = []

    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            skills_found.append(skill)

    return skills_found


def preprocess_resume(text: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns:
        cleaned text
        processed text
        extracted skills
    """

    cleaned = clean_text(text)

    # For embeddings we just use cleaned text
    processed = cleaned

    skills = extract_skills(text)

    return {
        "cleaned": cleaned,
        "processed": processed,
        "skills": skills
    }
