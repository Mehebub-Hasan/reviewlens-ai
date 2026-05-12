import os
import re
import string
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, AuthenticationError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from transformers import pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page setup
st.set_page_config(
    page_title="ReviewLens AI",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    .hero-card {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(0, 219, 222, 0.14), rgba(252, 0, 255, 0.10));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.8rem;
    }

    .main-title {
        font-size: 48px;
        font-weight: 850;
        letter-spacing: -1px;
        margin-bottom: 0.25rem;
        color: #FFFFFF;
    }

    .subtitle {
        font-size: 18px;
        color: #C8D0DA;
        line-height: 1.6;
        max-width: 900px;
        margin-top: 0.5rem;
    }

    .badge-row {
        margin-top: 1.1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.13);
        color: #E8EEF7;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-card {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }

    .metric-label {
        font-size: 13px;
        color: #AEB8C5;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
    }

    .metric-help {
        font-size: 12px;
        color: #8D98A7;
        margin-top: 0.2rem;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 23, 34, 1), rgba(14, 17, 25, 1));
    }

    /* Subtle lift on the metric cards. */
    .metric-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 38px rgba(0, 0, 0, 0.30);
    }

    .hero-tagline {
        margin-top: 1.1rem;
        font-size: 12.5px;
        color: #8FA0B4;
        letter-spacing: 0.2px;
    }

    /* Calmer, consistent dividers / horizontal rules. */
    hr {
        margin: 1.3rem 0;
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.10);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-card">
        <div class="main-title">ReviewLens AI</div>
        <div class="subtitle">
            Upload any customer-review or feedback CSV and get clustering, sentiment,
            key topics, interactive 2D/3D charts, and chatbot-powered answers &mdash; all in one pass.
        </div>
        <div class="badge-row">
            <span class="badge">TF-IDF</span>
            <span class="badge">TruncatedSVD</span>
            <span class="badge">K-Means</span>
            <span class="badge">Hierarchical</span>
            <span class="badge">DBSCAN</span>
            <span class="badge">Deep NLP Sentiment</span>
            <span class="badge">LLM Chatbot</span>
        </div>
        <div class="hero-tagline">Understand your customers beyond star ratings.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# NLTK setup

@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = setup_nltk()


# Deep NLP sentiment model

@st.cache_resource
def load_deep_sentiment_model():
    return pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

# DeepSeek API setup

load_dotenv()


def get_secret_value(key):
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets[key]
    except Exception:
        return None


def get_deepseek_client():
    api_key = get_secret_value("DEEPSEEK_API_KEY")

    if not api_key or api_key == "your_deepseek_api_key_here":
        return None

    # Give DeepSeek a generous-but-bounded timeout and a couple of retries —
    # the API can be slow under load, which otherwise surfaces as a bare
    # "Request timed out" in the AI Dataset Analyst.
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=120.0,
        max_retries=2
    )

 # Helper functions

def load_csv_safely(uploaded_file):
    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1"]

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
        except Exception:
            continue

    raise ValueError("Could not read the CSV file. Please check if the file is valid.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


def create_sentiment_from_rating(rating):
    try:
        rating = float(rating)
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
    except Exception:
        return "Unknown"


def extract_numeric_rating(series):
    return (
        series.astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )


def get_top_words_per_cluster(tfidf_matrix, labels, terms, top_n=10):
    result = {}
    labels = np.array(labels)

    for cluster in sorted(set(labels)):
        cluster_indices = np.where(labels == cluster)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        cluster_tfidf = np.asarray(cluster_tfidf).flatten()

        top_indices = cluster_tfidf.argsort()[::-1][:top_n]
        top_words = [terms[i] for i in top_indices]

        result[cluster] = ", ".join(top_words)

    return pd.DataFrame(
        [{"cluster": k, "top_words": v} for k, v in result.items()]
    )


def make_cluster_summary(df, cluster_col, text_column):
    summary = df.groupby(cluster_col).agg(
        count=(text_column, "count")
    ).reset_index()

    if "Rating_Number" in df.columns:
        rating_summary = df.groupby(cluster_col)["Rating_Number"].mean().round(2)
        summary["avg_rating"] = summary[cluster_col].map(rating_summary)

    if "Rating_Sentiment" in df.columns:
        sentiment_pct = pd.crosstab(
            df[cluster_col],
            df["Rating_Sentiment"],
            normalize="index"
        ) * 100

        sentiment_pct = sentiment_pct.round(2).reset_index()
        summary = summary.merge(sentiment_pct, on=cluster_col, how="left")

    if "Deep_Sentiment" in df.columns:
        processed_df = df[df["Deep_Sentiment"] != "Not Processed"]

        if not processed_df.empty:
            deep_pct = pd.crosstab(
                processed_df[cluster_col],
                processed_df["Deep_Sentiment"],
                normalize="index"
            ) * 100

            deep_pct = deep_pct.round(2).reset_index()

            deep_pct = deep_pct.rename(
                columns={
                    "Negative": "Deep_Negative_%",
                    "Neutral": "Deep_Neutral_%",
                    "Positive": "Deep_Positive_%"
                }
            )

            summary = summary.merge(deep_pct, on=cluster_col, how="left")

    return summary


def evaluate_clustering(X_data, labels):
    labels = np.array(labels)

    if -1 in labels:
        mask = labels != -1
        X_eval = X_data[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X_data
        labels_eval = labels

    if len(set(labels_eval)) <= 1 or len(labels_eval) <= 1:
        return {
            "Silhouette Score": None,
            "Davies-Bouldin Score": None,
            "Calinski-Harabasz Score": None
        }

    return {
        "Silhouette Score": round(silhouette_score(X_eval, labels_eval), 4),
        "Davies-Bouldin Score": round(davies_bouldin_score(X_eval, labels_eval), 4),
        "Calinski-Harabasz Score": round(calinski_harabasz_score(X_eval, labels_eval), 4)
    }


def run_deep_sentiment_analysis(work_df, text_column, deep_sentiment_limit):
    sentiment_model = load_deep_sentiment_model()

    texts = work_df[text_column].astype(str).tolist()
    texts_to_process = texts[:deep_sentiment_limit]

    deep_labels = []
    deep_scores = []

    batch_size = 16
    total_batches = max(1, (len(texts_to_process) + batch_size - 1) // batch_size)

    progress_bar = st.progress(0)

    for batch_index, i in enumerate(range(0, len(texts_to_process), batch_size)):
        batch_texts = texts_to_process[i:i + batch_size]

        results = sentiment_model(
            batch_texts,
            truncation=True,
            max_length=512
        )

        for result in results:
            deep_labels.append(result["label"].capitalize())
            deep_scores.append(result["score"])

        progress_bar.progress((batch_index + 1) / total_batches)

    work_df["Deep_Sentiment"] = "Not Processed"
    work_df["Deep_Sentiment_Score"] = np.nan

    work_df.loc[:len(deep_labels) - 1, "Deep_Sentiment"] = deep_labels
    work_df.loc[:len(deep_scores) - 1, "Deep_Sentiment_Score"] = deep_scores

    return work_df


def show_answer(answer):
    if isinstance(answer, pd.DataFrame) or isinstance(answer, pd.Series):
        st.dataframe(answer, width="stretch")
    elif isinstance(answer, list):
        st.write(", ".join(map(str, answer)))
    else:
        st.write(answer)


def metric_card(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Small presentation helpers (kept thin: just DRY for repeated dashboard UI) ---

def section(title, caption=None):
    """A section subheader, with an optional caption line underneath."""
    st.subheader(title)
    if caption:
        st.caption(caption)


def count_table(series, names, sort_index=False):
    """Turn a Series into a tidy 2-column counts DataFrame with the given column names."""
    counts = series.value_counts()
    if sort_index:
        counts = counts.sort_index()
    out = counts.reset_index()
    out.columns = names
    return out


def bar_chart(data, x=None, y=None, color=None, text=None, title=None,
              barmode=None, height=440, show_legend=True, **layout):
    """Build a styled Plotly bar chart; the caller renders it with st.plotly_chart."""
    fig = px.bar(data, x=x, y=y, color=color, text=text, title=title, barmode=barmode)
    if text is not None:
        fig.update_traces(textposition="outside")
    fig.update_layout(height=height, showlegend=show_legend, **layout)
    return fig


# The scatter plots are the heaviest figures and st.tabs renders every tab on
# each rerun, so cache them on the (already sampled, <=5000-row) plot frame.
# NOTE: we deliberately do NOT cache make_cluster_summary / evaluate_clustering /
# get_top_words_per_cluster / load_csv_safely — the first three only run inside the
# "Analyze" block (results live in st.session_state), and caching the CSV loader
# would break the file-uploader re-read behaviour.

@st.cache_data(show_spinner=False)
def make_cluster_scatter_2d(plot_df, text_column):
    """2D K-Means scatter over the first two SVD components."""
    fig = px.scatter(
        plot_df,
        x="SVD1",
        y="SVD2",
        color=plot_df["kmeans_cluster"].astype(str),
        hover_data=[text_column, "kmeans_cluster"],
        title="2D K-Means Cluster Visualization",
        opacity=0.65
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        legend_title_text="Cluster",
        height=650,
        xaxis_title="SVD Component 1",
        yaxis_title="SVD Component 2"
    )
    return fig


@st.cache_data(show_spinner=False)
def make_cluster_scatter_3d(plot_df, text_column):
    """3D K-Means scatter over the first three SVD components."""
    fig = px.scatter_3d(
        plot_df,
        x="SVD3_1",
        y="SVD3_2",
        z="SVD3_3",
        color=plot_df["kmeans_cluster"].astype(str),
        hover_data=[text_column, "kmeans_cluster"],
        title="3D K-Means Cluster Visualization",
        opacity=0.75
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend_title_text="Cluster",
        height=750,
        scene=dict(
            xaxis_title="SVD Component 1",
            yaxis_title="SVD Component 2",
            zaxis_title="SVD Component 3"
        )
    )
    return fig


# --- Rule-based dataset chatbot (no LLM, no exec — just lookups on the result) ---

def simple_chatbot(question, analyzed_df, top_words_df, evaluation_df, text_column):
    """Answer a small set of common questions about the analyzed dataframe."""
    q = question.lower()

    if "row" in q or "shape" in q or "size" in q:
        return f"The analyzed dataset has {analyzed_df.shape[0]} rows and {analyzed_df.shape[1]} columns."

    if "column" in q:
        return analyzed_df.columns.tolist()

    if "cluster size" in q or "cluster count" in q:
        return analyzed_df["kmeans_cluster"].value_counts().sort_index()

    if "top words" in q:
        return top_words_df

    if "average rating" in q and "Rating_Number" in analyzed_df.columns:
        return analyzed_df.groupby("kmeans_cluster")["Rating_Number"].mean().round(2)

    if "lowest rating" in q and "Rating_Number" in analyzed_df.columns:
        means = analyzed_df.groupby("kmeans_cluster")["Rating_Number"].mean()
        return f"Cluster {means.idxmin()} has the lowest average rating: {means.min():.2f}"

    if "highest rating" in q and "Rating_Number" in analyzed_df.columns:
        means = analyzed_df.groupby("kmeans_cluster")["Rating_Number"].mean()
        return f"Cluster {means.idxmax()} has the highest average rating: {means.max():.2f}"

    if "deep sentiment percentage" in q and "Deep_Sentiment" in analyzed_df.columns:
        processed = analyzed_df[analyzed_df["Deep_Sentiment"] != "Not Processed"]
        if processed.empty:
            return "Deep sentiment was not processed. Enable Deep NLP Sentiment Analysis before running the analysis."
        return (pd.crosstab(
            processed["kmeans_cluster"], processed["Deep_Sentiment"], normalize="index"
        ) * 100).round(2)

    if "deep sentiment" in q and "Deep_Sentiment" in analyzed_df.columns:
        processed = analyzed_df[analyzed_df["Deep_Sentiment"] != "Not Processed"]
        if processed.empty:
            return "Deep sentiment was not processed. Enable Deep NLP Sentiment Analysis before running the analysis."
        return pd.crosstab(processed["kmeans_cluster"], processed["Deep_Sentiment"])

    if "sentiment" in q and "Rating_Sentiment" in analyzed_df.columns:
        return pd.crosstab(analyzed_df["kmeans_cluster"], analyzed_df["Rating_Sentiment"])

    if "example" in q:
        return analyzed_df[[text_column, "kmeans_cluster"]].head(10)

    if "evaluation" in q or "score" in q:
        return evaluation_df

    return (
        "I can answer about rows, columns, cluster size, top words, average / highest / lowest rating, "
        "rating sentiment, deep sentiment, evaluation scores, and example texts."
    )


# DeepSeek LLM chatbot functions

def is_code_safe(code):
    banned_patterns = [
        r"\bimport\b",
        r"\bopen\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"__",
        r"\bos\b",
        r"\bsubprocess\b",
        r"\brequests\b",
        r"\bsocket\b",
        r"\bshutil\b",
        r"\bpathlib\b",
        r"\bto_csv\b",
        r"\bto_excel\b",
        r"\bto_pickle\b",
        r"\bread_csv\b",
        r"\bread_excel\b",
        r"\bremove\b",
        r"\bunlink\b",
        r"\brmdir\b",
        r"\bmkdir\b",
        r"\bsystem\b",
    ]

    for pattern in banned_patterns:
        if re.search(pattern, code):
            return False

    return "result" in code


def generate_pandas_code_with_deepseek(client, df, question):
    sample_rows = df.head(3).to_string(index=False)

    prompt = f"""
You are a data analyst assistant working with a Pandas DataFrame named df.

Dataset shape:
{df.shape}

Columns:
{df.columns.tolist()}

Sample rows:
{sample_rows}

User question:
{question}

Generate ONLY safe Python Pandas code to answer the question.

Rules:
- Use only the existing DataFrame named df.
- The final answer must be assigned to a variable named result.
- Do not import anything.
- Do not read or write files.
- Do not use open(), os, subprocess, eval(), exec(), requests, networking, hidden/private attributes, or file operations.
- Do not modify df permanently.
- Keep the code short and simple.
- Return code only.
- No markdown.
- No explanation.

Examples:
result = df["kmeans_cluster"].value_counts()
result = df.groupby("kmeans_cluster")["Rating_Number"].mean().round(2)
result = pd.crosstab(df["kmeans_cluster"], df["Rating_Sentiment"])
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You generate safe Pandas code only. Return code only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    return code


def run_generated_code(code, df):
    local_vars = {
        "df": df,
        "pd": pd,
        "np": np
    }

    exec(code, {"__builtins__": {}}, local_vars)

    if "result" not in local_vars:
        raise ValueError("Generated code did not create a result variable.")

    return local_vars["result"]


def explain_result_with_deepseek(client, question, code, result):
    result_text = str(result)

    if len(result_text) > 3000:
        result_text = result_text[:3000] + "\n...output truncated..."

    prompt = f"""
User question:
{question}

Pandas code used:
{code}

Result:
{result_text}

Explain the result briefly and clearly.
Mention important numbers or patterns.
Do not overclaim beyond the result.
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You explain data analysis results clearly and professionally."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()    


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("ReviewLens AI")
    st.caption("NLP & ML analysis for customer reviews and feedback.")

    st.markdown("### How it works")
    st.markdown(
        """
        1. Upload a CSV
        2. Pick the text column (rating column optional)
        3. TF-IDF &rarr; TruncatedSVD
        4. K-Means / Hierarchical / DBSCAN
        5. Rating-based + deep RoBERTa sentiment
        6. 2D / 3D visualizations
        7. Ask the dataset chatbots
        8. Download the analyzed CSV
        """
    )

    st.markdown("### Status")
    if get_deepseek_client() is not None:
        st.success("AI Dataset Analyst — configured")
    else:
        st.info("AI Dataset Analyst — not configured")
    st.caption(
        "The deep-sentiment model downloads on first use, so the first "
        "deep-sentiment run can take a minute."
    )



# -----------------------------
# Upload dataset
# -----------------------------

st.subheader("Upload your dataset")
st.caption(
    "Any CSV of customer reviews or feedback. You'll pick the text column next; "
    "a rating column is optional but unlocks rating-based sentiment."
)
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv_safely(uploaded_file)
    except Exception as e:
        st.error(f"Could not load CSV file: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded dataset has no rows.")
        st.stop()

    section("Dataset preview", f"First 10 of {df.shape[0]:,} rows.")
    col_a, col_b = st.columns(2)
    col_a.metric("Rows", f"{df.shape[0]:,}")
    col_b.metric("Columns", df.shape[1])
    st.dataframe(df.head(10), width="stretch")

    # -----------------------------
    # Analysis Settings Form
    # -----------------------------

    section(
        "Analysis settings",
        "Fast Mode for quick exploration · Advanced Mode for the full pipeline."
    )

    with st.form("analysis_settings_form"):
        text_column = st.selectbox(
            "Select the text/review/feedback column",
            df.columns
        )

        rating_column = st.selectbox(
            "Select rating column optional",
            ["None"] + list(df.columns)
        )

        analysis_mode = st.radio(
            "Select analysis mode",
            ["Fast Mode", "Advanced Mode"],
            horizontal=True
        )

        if analysis_mode == "Fast Mode":
            st.info(
                "Fast Mode is optimized for speed. It uses K-Means by default and limits heavy processing."
            )

            max_rows = st.number_input(
                "Maximum rows to analyze",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000
            )

            k = st.slider(
                "Select number of clusters",
                min_value=2,
                max_value=10,
                value=3
            )

            run_hclust = False
            run_dbscan = False

            run_deep_sentiment = st.checkbox(
                "Run Deep NLP Sentiment Analysis (RoBERTa model)",
                value=False
            )

            deep_sentiment_limit = st.number_input(
                "Deep sentiment row limit",
                min_value=100,
                max_value=3000,
                value=500,
                step=100
            )

        else:
            st.warning(
                "Advanced Mode runs the full pipeline. It may take longer on large datasets."
            )

            max_rows = st.number_input(
                "Maximum rows to analyze",
                min_value=1000,
                max_value=100000,
                value=20000,
                step=1000
            )

            k = st.slider(
                "Select number of clusters",
                min_value=2,
                max_value=10,
                value=3
            )

            run_hclust = st.checkbox("Run Hierarchical Clustering", value=True)
            run_dbscan = st.checkbox("Run DBSCAN", value=True)

            run_deep_sentiment = st.checkbox(
                "Run Deep NLP Sentiment Analysis (RoBERTa model)",
                value=False
            )

            deep_sentiment_limit = st.number_input(
                "Deep sentiment row limit",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )

        st.caption(
            "Fast Mode is recommended for smooth interaction. Advanced Mode is best for full analysis."
        )

        analyze_button = st.form_submit_button(
            "Analyze Dataset",
            type="primary"
        )

    # -----------------------------
    # Run Analysis
    # -----------------------------

    if analyze_button:
        with st.spinner("Analyzing dataset..."):
            work_df = df.copy()

            # Apply row limit for smoother performance
            if work_df.shape[0] > int(max_rows):
                work_df = work_df.sample(
                    n=int(max_rows),
                    random_state=123
                ).reset_index(drop=True)

                st.info(
                    f"Dataset was sampled to {int(max_rows):,} rows for {analysis_mode}."
                )

            # Remove missing and short text
            work_df = work_df.dropna(subset=[text_column])
            work_df["word_count"] = work_df[text_column].apply(
                lambda x: len(str(x).split())
            )
            work_df = work_df[work_df["word_count"] >= 3]
            work_df = work_df.reset_index(drop=True)

            if work_df.shape[0] < k:
                st.error("Not enough valid text rows for the selected number of clusters.")
                st.stop()

            # Clean text
            work_df["Cleaned_Text"] = work_df[text_column].apply(clean_text)
            work_df = work_df[work_df["Cleaned_Text"].str.strip() != ""]
            work_df = work_df.reset_index(drop=True)

            if work_df.shape[0] < k:
                st.error("After text cleaning, not enough valid rows remain for clustering.")
                st.stop()

            # Rating-based sentiment
            if rating_column != "None":
                try:
                    work_df["Rating_Number"] = extract_numeric_rating(
                        work_df[rating_column]
                    )
                    work_df["Rating_Sentiment"] = work_df["Rating_Number"].apply(
                        create_sentiment_from_rating
                    )
                except Exception:
                    st.warning(
                        "Could not extract numeric ratings. Rating-based sentiment was skipped."
                    )

            # TF-IDF
            tfidf = TfidfVectorizer(
                max_features=3000,
                min_df=2,
                max_df=0.85,
                ngram_range=(1, 2)
            )

            try:
                tfidf_matrix = tfidf.fit_transform(work_df["Cleaned_Text"])
            except ValueError:
                st.error("TF-IDF failed. The text column may not contain enough meaningful words.")
                st.stop()

            if tfidf_matrix.shape[1] < 2:
                st.error("Not enough unique text features for analysis.")
                st.stop()

            # SVD features for clustering
            n_components = min(
                50,
                tfidf_matrix.shape[1] - 1,
                work_df.shape[0] - 1
            )

            if n_components < 2:
                st.error("Not enough data for dimensionality reduction.")
                st.stop()

            svd = TruncatedSVD(n_components=n_components, random_state=123)
            X = svd.fit_transform(tfidf_matrix)

            # 2D and 3D SVD for visualization
            svd_2d = TruncatedSVD(n_components=2, random_state=123)
            X_2d = svd_2d.fit_transform(tfidf_matrix)

            svd_3d = TruncatedSVD(n_components=3, random_state=123)
            X_3d = svd_3d.fit_transform(tfidf_matrix)

            work_df["SVD1"] = X_2d[:, 0]
            work_df["SVD2"] = X_2d[:, 1]
            work_df["SVD3_1"] = X_3d[:, 0]
            work_df["SVD3_2"] = X_3d[:, 1]
            work_df["SVD3_3"] = X_3d[:, 2]

            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=123, n_init=10)
            work_df["kmeans_cluster"] = kmeans.fit_predict(X)

            # Hierarchical Clustering
            if run_hclust:
                hclust = AgglomerativeClustering(n_clusters=k)
                work_df["hclust_cluster"] = hclust.fit_predict(X)
            else:
                work_df["hclust_cluster"] = -1

            # DBSCAN Clustering
            X20 = X[:, :min(20, X.shape[1])]

            if run_dbscan:
                dbscan = DBSCAN(eps=1.5, min_samples=5)
                work_df["dbscan_cluster"] = dbscan.fit_predict(X20)
            else:
                work_df["dbscan_cluster"] = -1

            # Deep NLP sentiment
            if run_deep_sentiment:
                with st.spinner("Running deep NLP sentiment analysis..."):
                    work_df = run_deep_sentiment_analysis(
                        work_df,
                        text_column,
                        int(deep_sentiment_limit)
                    )

            # Summaries
            kmeans_summary = make_cluster_summary(
                work_df,
                "kmeans_cluster",
                text_column
            )

            hclust_summary = make_cluster_summary(
                work_df,
                "hclust_cluster",
                text_column
            )

            dbscan_summary = make_cluster_summary(
                work_df,
                "dbscan_cluster",
                text_column
            )

            # Evaluations
            evaluation_rows = [
                {
                    "Model": "K-Means",
                    **evaluate_clustering(X, work_df["kmeans_cluster"])
                }
            ]

            if run_hclust:
                evaluation_rows.append(
                    {
                        "Model": "Hierarchical",
                        **evaluate_clustering(X, work_df["hclust_cluster"])
                    }
                )

            if run_dbscan:
                evaluation_rows.append(
                    {
                        "Model": "DBSCAN",
                        **evaluate_clustering(X20, work_df["dbscan_cluster"])
                    }
                )

            evaluation_df = pd.DataFrame(evaluation_rows)

            # Top words
            terms = tfidf.get_feature_names_out()

            top_words_df = get_top_words_per_cluster(
                tfidf_matrix,
                work_df["kmeans_cluster"].values,
                terms,
                top_n=10
            )

            # Save to session state
            st.session_state["analyzed_df"] = work_df
            st.session_state["kmeans_summary"] = kmeans_summary
            st.session_state["hclust_summary"] = hclust_summary
            st.session_state["dbscan_summary"] = dbscan_summary
            st.session_state["evaluation_df"] = evaluation_df
            st.session_state["top_words_df"] = top_words_df
            st.session_state["text_column"] = text_column
            st.session_state["analysis_mode"] = analysis_mode
            st.session_state["run_hclust"] = run_hclust
            st.session_state["run_dbscan"] = run_dbscan

        st.success("Analysis completed successfully!")

else:
    if "analyzed_df" in st.session_state:
        st.info(
            "Showing your most recent analysis below. "
            "Upload a new CSV above to run a fresh one."
        )
    else:
        st.info(
            "Upload a CSV review or feedback dataset above to get started — "
            "include a free-text review column, and a rating column if you have one."
        )


# Results

if "analyzed_df" in st.session_state:
    analyzed_df = st.session_state["analyzed_df"]
    kmeans_summary = st.session_state["kmeans_summary"]
    hclust_summary = st.session_state["hclust_summary"]
    dbscan_summary = st.session_state["dbscan_summary"]
    evaluation_df = st.session_state["evaluation_df"]
    top_words_df = st.session_state["top_words_df"]
    text_column = st.session_state["text_column"]
    analysis_mode = st.session_state.get("analysis_mode", "Fast Mode")
    run_hclust = st.session_state.get("run_hclust", False)
    run_dbscan = st.session_state.get("run_dbscan", False)

    st.divider()
    st.header("Analysis dashboard")
    st.caption(f"Mode: {analysis_mode}  ·  {analyzed_df.shape[0]:,} reviews analyzed")

    dbscan_groups = analyzed_df["dbscan_cluster"].nunique()
    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        metric_card(
            "Reviews analyzed",
            f"{analyzed_df.shape[0]:,}",
            "Valid review texts after cleaning"
        )

    with metric2:
        metric_card(
            "Output columns",
            analyzed_df.shape[1],
            "Original columns + analysis columns added"
        )

    with metric3:
        metric_card(
            "K-Means clusters",
            analyzed_df["kmeans_cluster"].nunique(),
            "Topic groups from K-Means"
        )

    with metric4:
        metric_card(
            "DBSCAN groups",
            dbscan_groups if run_dbscan else "—",
            "Includes a noise / outlier group" if run_dbscan else "DBSCAN not run"
        )

    # NOTE: st.tabs doesn't preserve the active tab across reruns — submitting a
    # chatbot question (or any widget interaction) kept bouncing users back to the
    # first tab. So the dashboard sections are driven by a session-state-backed
    # radio selector instead, which survives reruns via key="active_section".
    nav = st.radio(
        "Dashboard section",
        ["Cluster Summary", "Visualizations", "Sentiment", "Dataset Chatbot", "Download"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_section"
    )
    st.write("")


    # --- Section: Cluster Summary ---
    if nav == "Cluster Summary":
        section(
            "Clustering Quality",
            "Higher Silhouette / Calinski-Harabasz and lower Davies-Bouldin indicate tighter, "
            "better-separated clusters. Blank cells mean the metric couldn't be computed."
        )
        st.dataframe(evaluation_df, width="stretch", hide_index=True)

        section(
            "K-Means Cluster Summary",
            "Size of each cluster, plus average rating and sentiment mix where those columns exist."
        )
        st.dataframe(kmeans_summary, width="stretch", hide_index=True)

        if run_hclust:
            section("Hierarchical Cluster Summary")
            st.dataframe(hclust_summary, width="stretch", hide_index=True)
        else:
            st.info(
                "Hierarchical Clustering was not run for this analysis "
                "(it's off in Fast Mode and optional in Advanced Mode)."
            )

        if run_dbscan:
            section("DBSCAN Cluster Summary", "Cluster -1 is the noise / outlier group.")
            st.dataframe(dbscan_summary, width="stretch", hide_index=True)
        else:
            st.info(
                "DBSCAN was not run for this analysis "
                "(it's off in Fast Mode and optional in Advanced Mode)."
            )

        section(
            "Top Words Per K-Means Cluster",
            "The highest-weighted TF-IDF terms in each cluster — a quick read on what each cluster is about."
        )
        st.dataframe(top_words_df, width="stretch", hide_index=True)

        section("Example Texts by K-Means Cluster")
        selected_cluster = st.selectbox(
            "Pick a cluster to see sample reviews",
            sorted(analyzed_df["kmeans_cluster"].unique())
        )
        examples = analyzed_df[analyzed_df["kmeans_cluster"] == selected_cluster][
            [text_column, "kmeans_cluster"]
        ].head(10)
        st.dataframe(examples, width="stretch")

    # --- Section: Visualizations ---
    elif nav == "Visualizations":
        section(
            "K-Means Cluster Size",
            "How many reviews fell into each K-Means cluster."
        )

        cluster_counts = count_table(
            analyzed_df["kmeans_cluster"], ["Cluster", "Count"], sort_index=True
        )
        st.dataframe(cluster_counts, width="stretch", hide_index=True)

        st.plotly_chart(
            bar_chart(
                cluster_counts,
                x="Cluster",
                y="Count",
                color="Cluster",
                text="Count",
                title="K-Means Cluster Size",
                height=500,
                show_legend=False,
                xaxis_title="Cluster",
                yaxis_title="Number of Reviews"
            ),
            width="stretch"
        )

        # Sample down before the scatter plots so large datasets stay responsive.
        plot_limit = 5000
        if analyzed_df.shape[0] > plot_limit:
            plot_df = analyzed_df.sample(plot_limit, random_state=123)
            st.info(f"Showing {plot_limit:,} sampled rows for faster 2D/3D visualization.")
        else:
            plot_df = analyzed_df.copy()

        # --- 2D scatter: first two SVD components ---
        section(
            "2D K-Means Cluster Visualization",
            "Each point is a review, placed by its first two SVD components and coloured by cluster."
        )
        st.plotly_chart(
            make_cluster_scatter_2d(plot_df, text_column), width="stretch"
        )

        # --- 3D scatter: first three SVD components ---
        section("3D K-Means Cluster Visualization")
        st.info(
            "Clustering runs on higher-dimensional SVD features; this chart shows only "
            "the first 3 components, so small or outlier clusters can look tiny in 3D."
        )
        st.plotly_chart(
            make_cluster_scatter_3d(plot_df, text_column), width="stretch"
        )

    
    # --- Section: Sentiment ---
    elif nav == "Sentiment":
        if "Rating_Sentiment" in analyzed_df.columns:
            section(
                "Rating-Based Sentiment Distribution",
                "Sentiment derived from the numeric rating column you selected."
            )
            sentiment_counts = count_table(
                analyzed_df["Rating_Sentiment"], ["Sentiment", "Count"]
            )
            st.plotly_chart(
                bar_chart(
                    sentiment_counts,
                    x="Sentiment",
                    y="Count",
                    text="Count",
                    title="Rating-Based Sentiment Distribution"
                ),
                width="stretch"
            )

            section("Rating-Based Sentiment by K-Means Cluster")
            sentiment_cluster = pd.crosstab(
                analyzed_df["kmeans_cluster"],
                analyzed_df["Rating_Sentiment"]
            )
            st.dataframe(sentiment_cluster, width="stretch")
            st.plotly_chart(
                bar_chart(
                    sentiment_cluster,
                    barmode="group",
                    title="Rating-Based Sentiment by K-Means Cluster"
                ),
                width="stretch"
            )

            if "Rating_Number" in analyzed_df.columns:
                section("Average Rating by K-Means Cluster")
                avg_rating = (
                    analyzed_df.groupby("kmeans_cluster")["Rating_Number"]
                    .mean().round(2).reset_index()
                )
                st.plotly_chart(
                    bar_chart(
                        avg_rating,
                        x="kmeans_cluster",
                        y="Rating_Number",
                        text="Rating_Number",
                        title="Average Rating by K-Means Cluster",
                        xaxis_title="K-Means Cluster",
                        yaxis_title="Average Rating"
                    ),
                    width="stretch"
                )
        else:
            st.info("No rating column was selected, so rating-based sentiment is not available.")

        if "Deep_Sentiment" in analyzed_df.columns:
            st.markdown("---")
            section(
                "Deep NLP Sentiment Distribution",
                "Sentiment predicted by the RoBERTa model on the review text."
            )
            deep_sentiment_counts = count_table(
                analyzed_df["Deep_Sentiment"], ["Deep Sentiment", "Count"]
            )
            st.plotly_chart(
                bar_chart(
                    deep_sentiment_counts,
                    x="Deep Sentiment",
                    y="Count",
                    text="Count",
                    title="Deep NLP Sentiment Distribution"
                ),
                width="stretch"
            )

            processed_deep_df = analyzed_df[analyzed_df["Deep_Sentiment"] != "Not Processed"]

            if not processed_deep_df.empty:
                section("Deep NLP Sentiment by K-Means Cluster")
                deep_cluster = pd.crosstab(
                    processed_deep_df["kmeans_cluster"],
                    processed_deep_df["Deep_Sentiment"]
                )
                st.dataframe(deep_cluster, width="stretch")
                st.plotly_chart(
                    bar_chart(
                        deep_cluster,
                        barmode="group",
                        title="Deep NLP Sentiment by K-Means Cluster"
                    ),
                    width="stretch"
                )

                section(
                    "Deep NLP Sentiment Percentage by Cluster",
                    "Each row sums to 100% across the sentiment labels."
                )
                deep_cluster_pct = pd.crosstab(
                    processed_deep_df["kmeans_cluster"],
                    processed_deep_df["Deep_Sentiment"],
                    normalize="index"
                ) * 100
                st.dataframe(deep_cluster_pct.round(2), width="stretch")
            else:
                st.info(
                    "Deep sentiment columns exist but no rows were processed yet. "
                    "Re-run the analysis with a higher deep-sentiment row limit."
                )

        
    # --- Section: Dataset Chatbot ---
    elif nav == "Dataset Chatbot":
        section(
            "Basic dataset chatbot",
            "Quick, rule-based answers — no AI, runs entirely on your analyzed data."
        )
        with st.expander("Example questions"):
            st.markdown(
                "- Show cluster size\n"
                "- Show top words\n"
                "- Show average rating by cluster\n"
                "- Which cluster has the lowest / highest rating?\n"
                "- Show sentiment by cluster\n"
                "- Show deep sentiment / deep sentiment percentage\n"
                "- Show example reviews\n"
                "- Show evaluation scores"
            )

        with st.form("basic_chatbot_form"):
            basic_question = st.text_input(
                "Ask a basic question",
                placeholder="e.g. show average rating by cluster"
            )
            basic_submitted = st.form_submit_button("Ask")

        if basic_submitted and basic_question.strip():
            section("Answer")
            show_answer(
                simple_chatbot(
                    basic_question, analyzed_df, top_words_df, evaluation_df, text_column
                )
            )
        elif basic_submitted:
            st.caption("Type a question above, then press Ask.")
        else:
            st.caption("Ask a question above for an instant answer.")

        st.markdown("---")

        # --- AI Dataset Analyst (LLM-powered) ---
        section(
            "AI Dataset Analyst",
            "Ask in plain English. The AI analyst converts your question into safe analysis "
            "logic, runs it on your data, and explains the result."
        )
        with st.expander("Example questions for the AI analyst"):
            st.markdown(
                "- Which cluster has the lowest average rating, and what does that suggest?\n"
                "- Compare sentiment across clusters.\n"
                "- Which value in a given column has the most reviews?\n"
                "- Give me 5 example negative reviews from the worst cluster.\n"
                "- What business insight can we draw from this dataset?"
            )

        deepseek_client = get_deepseek_client()

        if deepseek_client is None:
            st.warning(
                "The AI Dataset Analyst needs an API key. Locally, add `DEEPSEEK_API_KEY` "
                "to a `.env` file; on Streamlit Community Cloud, add it under App settings → "
                "Secrets. The basic chatbot above works without it."
            )
        else:
            with st.form("ai_analyst_form"):
                llm_question = st.text_input(
                    "Ask the AI analyst",
                    key="deepseek_question",
                    placeholder="e.g. which cluster has the lowest average rating and why?"
                )
                llm_submitted = st.form_submit_button("Ask the AI analyst")

            if llm_submitted and llm_question.strip():
                try:
                    with st.spinner("Writing the analysis..."):
                        generated_code = generate_pandas_code_with_deepseek(
                            deepseek_client, analyzed_df, llm_question
                        )

                    if not is_code_safe(generated_code):
                        st.error(
                            "The generated analysis was blocked by the safety check. "
                            "Please rephrase your question."
                        )
                    else:
                        with st.spinner("Running it on your dataset..."):
                            llm_result = run_generated_code(generated_code, analyzed_df)

                        # The written explanation is a nice-to-have — if the AI
                        # service is slow/unavailable for this step, still show the result.
                        explanation = None
                        try:
                            with st.spinner("Explaining the result..."):
                                explanation = explain_result_with_deepseek(
                                    deepseek_client, llm_question, generated_code, llm_result
                                )
                        except Exception:
                            explanation = None

                        section("Answer")
                        if explanation:
                            st.write(explanation)
                            with st.expander("Show the raw result"):
                                show_answer(llm_result)
                        else:
                            st.caption("The AI couldn't write a summary this time — here's the raw result:")
                            show_answer(llm_result)

                        with st.expander("Show the generated analysis code"):
                            st.code(generated_code, language="python")
                except (APITimeoutError, APIConnectionError):
                    st.error(
                        "The AI service is slow or unreachable right now. Please try again "
                        "in a moment — the basic chatbot above still works in the meantime."
                    )
                except AuthenticationError:
                    st.error(
                        "The API key was rejected. Double-check `DEEPSEEK_API_KEY` in your "
                        "`.env` file (or in Streamlit Secrets when deployed)."
                    )
                except Exception as e:
                    st.error(f"AI analyst error: {e}")
            elif llm_submitted:
                st.caption("Type a question above, then press Ask the AI analyst.")
            else:
                st.caption("Ask a question above for an AI-written analysis.")
    
    # --- Section: Download ---
    elif nav == "Download":
        section(
            "Download analyzed dataset",
            "The full dataset plus every column the pipeline added — cleaned text, "
            "SVD components, cluster labels, sentiment, and so on."
        )

        csv_data = analyzed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download analyzed CSV",
            data=csv_data,
            file_name="reviewlens_analyzed_dataset.csv",
            mime="text/csv",
            type="primary"
        )
        st.caption(f"{analyzed_df.shape[0]:,} rows × {analyzed_df.shape[1]} columns")

        section("Preview", "First 20 rows of the analyzed dataset.")
        st.dataframe(analyzed_df.head(20), width="stretch")