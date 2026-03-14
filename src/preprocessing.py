import re
import string
import spacy


nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
 
    if not isinstance(text, str):
        return ""

    
    text = text.lower()

    
    text = re.sub(r"https?:\/\/\S+|www\.\S+", " ", text)

    
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    
    text = text.translate(str.maketrans("", "", string.punctuation))

    
    text = re.sub(r"\s+", " ", text)

    
    return text.strip()


def lemmatize_text(text: str) -> str:

    doc = nlp(text)

   
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]

    return " ".join(lemmas)


def extract_skills(text: str) -> list:
    doc = nlp(text)

    skills = set()

    
    for entity in doc.ents:
        if entity.label_ in ("ORG", "PRODUCT", "GPE", "LANGUAGE"):
            skills.add(entity.text.lower())

    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:  # Skip very long phrases
            skills.add(chunk.text.lower())

    return list(skills)


def preprocess_resume(text: str, lemmatize: bool = True) -> dict:
    """
    Full preprocessing pipeline for a single resume.

    This is the main function called by other modules.
    It returns a dictionary so callers can use whichever
    representation they need.

    Args:
        text:      Raw resume text.
        lemmatize: Whether to also return lemmatized version.

    Returns:
        dict with keys:
            "cleaned"    → cleaned text (used for display)
            "processed"  → cleaned + optionally lemmatized (used for embedding)
            "skills"     → list of extracted skills
    """
    cleaned = clean_text(text)

    if lemmatize:
        processed = lemmatize_text(cleaned)
    else:
        processed = cleaned

    skills = extract_skills(text)  # Use original text for better NER

    return {
        "cleaned": cleaned,
        "processed": processed,
        "skills": skills,
    }