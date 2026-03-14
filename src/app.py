import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ranking_engine import rank_candidates, index_all_resumes
from src.pinecone_index import get_pinecone_client, get_index, get_index_stats


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
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .score-high   { color: #16a34a; font-weight: 600; }
    .score-medium { color: #ca8a04; font-weight: 600; }
    .score-low    { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def score_color_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    return "score-low"


@st.cache_resource
def get_pinecone_index():
    pc = get_pinecone_client()
    index = get_index(pc)
    return index


with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider(
        label="Number of candidates to return",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Higher values show more candidates but may include lower-quality matches."
    )

    st.divider()

    st.markdown("### Filter by category")
    categories = ["All", "Engineering", "Accounting", "Banking", "Teacher",
                  "HR", "Healthcare", "Sales", "Marketing"]
    selected_category = st.selectbox("Profession category", categories)

    st.divider()

    st.markdown("### Index management")
    st.caption("Run indexing when you first set up the system or add new resumes.")

    if st.button("🔄 Re-index all resumes", use_container_width=True):
        with st.spinner("Indexing resumes... this may take a few minutes."):
            index_all_resumes()
        st.success("✅ Indexing complete!")

    try:
        index = get_pinecone_index()
        stats = get_index_stats(index)
        st.info(f"📊 **{stats['total_vector_count']}** resumes indexed")
    except Exception:
        st.warning("⚠️ Could not connect to Pinecone.")


st.markdown('<div class="main-header">🔍 AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Powered by Sentence Transformers + Pinecone vector search</div>',
    unsafe_allow_html=True,
)

col_input, col_help = st.columns([2, 1])

with col_input:
    st.markdown("### Job Description")
    job_description = st.text_area(
        label="Paste the job description below:",
        placeholder=(
            "Example:\n\n"
            "We are looking for a Senior Data Engineer with 5+ years of experience "
            "in Python, Apache Spark, and cloud platforms (AWS/GCP). The ideal candidate "
            "has strong experience with ETL pipelines, data warehousing, and SQL optimization. "
            "Experience with machine learning workflows is a plus."
        ),
        height=200,
        label_visibility="collapsed",
    )

with col_help:
    st.markdown("### Tips for better results")
    st.markdown("""
    - **Be specific**: Include required skills, years of experience, and domain keywords.
    - **Use natural language**: Write it like a real job posting.
    - **Include must-haves**: Hard requirements produce sharper results.
    - **Adjust top-K**: Use the sidebar slider to control how many candidates to show.
    """)

_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    search_clicked = st.button(
        "🔍 Find Candidates",
        use_container_width=True,
        type="primary",
    )


if search_clicked:
    if not job_description.strip():
        st.error("Please enter a job description before searching.")
    else:
        with st.spinner("Analyzing job description and searching for candidates..."):
            results_df = rank_candidates(job_description, top_k=top_k)

        if results_df.empty:
            st.warning("No candidates found. Try broadening your job description.")
        else:
            if selected_category != "All":
                results_df = results_df[
                    results_df["category"].str.lower() == selected_category.lower()
                ]

            st.divider()
            st.markdown(f"### 🏆 Top {len(results_df)} Candidates Found")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total candidates", len(results_df))
            m2.metric("Best match score", f"{results_df['score'].max():.2%}")
            m3.metric("Average score", f"{results_df['score'].mean():.2%}")
            m4.metric("Categories found", results_df["category"].nunique())

            st.divider()

            for _, row in results_df.iterrows():
                score_pct = f"{row['score']:.1%}"

                if row["score"] >= 0.7:
                    score_emoji = "🟢"
                elif row["score"] >= 0.5:
                    score_emoji = "🟡"
                else:
                    score_emoji = "🔴"

                with st.expander(
                    f"#{row['rank']} — {row['category']} | {score_emoji} {score_pct} match",
                    expanded=(row["rank"] <= 3),
                ):
                    card_col1, card_col2 = st.columns([3, 1])

                    with card_col1:
                        st.markdown("**Resume preview:**")
                        st.caption(row["text_preview"][:500] + "..." if len(row["text_preview"]) > 500 else row["text_preview"])

                        if row["skills"]:
                            st.markdown("**Extracted skills:**")
                            skills_list = [s.strip() for s in row["skills"].split(",") if s.strip()]
                            skill_html = " ".join([
                                f'<span style="background:#e0f2fe;color:#0369a1;padding:2px 8px;'
                                f'border-radius:12px;font-size:12px;margin:2px;display:inline-block">'
                                f'{s}</span>'
                                for s in skills_list[:15]
                            ])
                            st.markdown(skill_html, unsafe_allow_html=True)

                    with card_col2:
                        st.markdown("**Match score:**")
                        st.progress(row["score"])
                        st.markdown(f"**{score_pct}**")
                        st.caption(f"Category: {row['category']}")
                        st.caption(f"ID: `{row['candidate_id'][:30]}...`")

            with st.expander("📊 View raw data table"):
                st.dataframe(
                    results_df[["rank", "category", "score", "match_pct", "skills"]],
                    use_container_width=True,
                )
                st.download_button(
                    label="⬇️ Download results as CSV",
                    data=results_df.to_csv(index=False),
                    file_name="top_candidates.csv",
                    mime="text/csv",
                )