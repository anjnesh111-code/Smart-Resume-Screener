import streamlit as st
import pandas as pd
import sys
import os
import uuid
import pdfplumber

# Fix import path so "src.xxx" imports work when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking_engine import rank_candidates, index_all_resumes
from src.pinecone_index import get_pinecone_client, get_index, get_index_stats
from src.embedding_model import get_embedding


st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------
# RESUME UPLOAD PIPELINE
# -----------------------------
def upload_resume(file, index) -> str:
    import hashlib
    text = extract_text_from_pdf(file)
    # Use content hash as ID so same resume never duplicates
    resume_id = "uploaded_" + hashlib.md5(text.encode()).hexdigest()
    embedding = get_embedding(text)
    index.upsert(vectors=[{
        "id":       resume_id,
        "values":   embedding,
        "metadata": {"text_preview": text[:1000], "category": "Uploaded", "skills": ""},
    }])
    return text


# -----------------------------
# PINECONE CONNECTION (cached)
# -----------------------------
@st.cache_resource
def get_pinecone_index():
    pc = get_pinecone_client()
    index = get_index(pc)
    return index


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider(
        label="Number of candidates to return",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.divider()
    st.markdown("### Filter by category")

    categories = [
        "All", "Engineering", "Accounting",
        "Banking", "Teacher", "HR",
        "Healthcare", "Sales", "Marketing",
    ]
    selected_category = st.selectbox("Profession category", categories)

    st.divider()
    st.markdown("### 📄 Upload Resume")

    uploaded_resume = st.file_uploader("Upload resume (PDF)", type=["pdf"])

    if uploaded_resume:
        try:
            index = get_pinecone_index()
            with st.spinner("Processing resume..."):
                text = upload_resume(uploaded_resume, index)
            st.success("✅ Resume added to vector database!")
            with st.expander("Resume Preview"):
                st.write(text[:1000])
        except Exception as e:
            st.error(f"❌ Failed to upload resume: {e}")

    st.divider()
    st.markdown("### Index management")

    if st.button("🔄 Re-index all resumes"):
        with st.spinner("Indexing resumes..."):
            try:
                index_all_resumes()
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")

    try:
        index = get_pinecone_index()
        stats = get_index_stats(index)
        st.info(f"📊 {stats['total_vector_count']} resumes indexed")
    except Exception as e:
        st.warning(f"⚠️ Could not connect to Pinecone: {e}")


# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-header">🔍 AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by Sentence Transformers + Pinecone Vector Search</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# JOB DESCRIPTION INPUT
# -----------------------------
col_input, col_help = st.columns([2, 1])

with col_input:
    st.markdown("### Job Description")
    job_description = st.text_area(
        label="Paste job description",
        height=200,
        label_visibility="collapsed",
    )

with col_help:
    st.markdown("### Tips")
    st.markdown("""
    • Include required skills  
    • Mention years of experience  
    • Use realistic job descriptions  
    """)


# -----------------------------
# SEARCH BUTTON
# -----------------------------
_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    search_clicked = st.button("🔍 Find Candidates", use_container_width=True)


# -----------------------------
# SEARCH RESULTS
# -----------------------------
if search_clicked:
    if not job_description.strip():
        st.error("Please enter a job description.")
    else:
        try:
            with st.spinner("Searching candidates..."):
                results_df = rank_candidates(job_description, top_k=top_k)

            if results_df.empty:
                st.warning("No candidates found. Make sure resumes have been indexed.")
            else:
                if selected_category != "All":
                    results_df = results_df[
                        results_df["category"].str.lower() == selected_category.lower()
                    ]

                st.divider()
                st.markdown(f"### 🏆 Top {len(results_df)} Candidates")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Candidates", len(results_df))
                m2.metric("Best score", f"{results_df['score'].max():.2%}")
                m3.metric("Average score", f"{results_df['score'].mean():.2%}")
                m4.metric("Categories", results_df["category"].nunique())

                st.divider()

                for _, row in results_df.iterrows():
                    score_pct = f"{row['score']:.1%}"
                    with st.expander(f"#{row['rank']} — {row['category']} | {score_pct} match"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown("**Resume Preview**")
                            st.write(str(row["text_preview"])[:500])
                            if row["skills"]:
                                st.markdown("**Skills**")
                                st.write(row["skills"])
                        with col2:
                            st.markdown("**Score**")
                            st.progress(float(min(max(row["score"], 0.0), 1.0)))
                            st.write(score_pct)
                            st.caption(f"Category: {row['category']}")
                            st.caption(f"ID: {str(row['candidate_id'])[:20]}")

                with st.expander("📊 Raw Data"):
                    st.dataframe(results_df)
                    st.download_button(
                        "Download CSV",
                        results_df.to_csv(index=False),
                        "candidates.csv",
                    )

        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            st.info("Make sure your PINECONE_API_KEY is set in Streamlit secrets.")
