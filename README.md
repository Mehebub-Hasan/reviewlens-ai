# ReviewLens AI

A deployable Streamlit app for analysing customer-review / feedback datasets end to end. Upload any CSV, pick the text column (and optionally a rating column), and ReviewLens AI runs a full NLP/ML pipeline and presents the results in a dashboard.

## What it does

- **Text cleaning & preprocessing** — lowercasing, URL/number/punctuation stripping, stop-word removal, lemmatization
- **TF-IDF** feature extraction → **TruncatedSVD** dimensionality reduction
- **Clustering** — K-Means, Hierarchical (Agglomerative), and DBSCAN, with Silhouette / Davies-Bouldin / Calinski-Harabasz evaluation
- **Top terms** per cluster and **example reviews** per cluster
- **Sentiment** — rating-based sentiment (from your rating column) and deep NLP sentiment via a RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Visualizations** — interactive 2D & 3D Plotly scatter plots of the SVD space, cluster-size and sentiment bar charts (large datasets are sampled for responsiveness)
- **Download** the fully analyzed dataset as CSV
- **Basic dataset chatbot** — instant rule-based answers (no AI, runs on your analyzed data)
- **AI Dataset Analyst** — LLM-powered: turns a plain-English question into safe analysis code, runs it on the analyzed data, and explains the result
- **Fast Mode** (quick exploration: K-Means only, lighter limits) and **Advanced Mode** (full pipeline incl. Hierarchical + DBSCAN)

## Project layout

```
app.py                  # the entire app (single file)
requirements.txt        # Python dependencies (version floors)
.streamlit/config.toml  # locks the dark theme (committed; no secrets here)
.gitignore              # ignores .env, secrets, caches, backups, OS junk
sample_data/            # (empty) drop sample CSVs here if you like
```

## Setup

Requires Python 3.10+ (tested on 3.13).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the NLTK corpora and (on the first deep-sentiment run) the RoBERTa model — so the first deep run takes a minute.

## Run

```bash
streamlit run app.py
```

## API key for the AI Dataset Analyst (optional)

Everything except the AI Dataset Analyst works without a key. The analyst is powered by an LLM — currently the DeepSeek API. To enable it:

**Locally** — create a `.env` file in the project root (it's git-ignored):

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**On Streamlit Community Cloud** — add it under **App settings → Secrets**:

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
```

The key is read via `python-dotenv` locally and `st.secrets` on Cloud — it is never printed or committed. The sidebar shows only whether it's configured.

> The generated analysis code is validated against a banned-pattern list and executed in a restricted namespace (no builtins, no file/network access). Treat it as a convenience, not a security boundary — don't point it at sensitive data.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the `.gitignore` keeps `.env` and `secrets.toml` out).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from the repo, main file `app.py`, Python 3.13.
3. Add `DEEPSEEK_API_KEY` under **Secrets** if you want the AI Dataset Analyst.
4. Deploy. `requirements.txt` and `.streamlit/config.toml` are picked up automatically.

## Notes

- `torch` + `transformers` are large; if you hit Streamlit Cloud resource limits, consider pinning a CPU-only PyTorch wheel.
- The whole heavy pipeline runs only when you click **Analyze Dataset**; results are cached in the session, so browsing the dashboard and chatting don't recompute anything.
