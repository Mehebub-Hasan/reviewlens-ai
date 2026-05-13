---
title: ReviewLens AI
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ReviewLens AI

**ReviewLens AI** is an AI-powered customer review analytics web app that helps users understand review datasets beyond simple star ratings.

Users can upload their own CSV review or feedback dataset, select the review text and rating columns, and automatically generate clustering insights, sentiment analysis, interactive visualizations, downloadable outputs, and chatbot-powered dataset Q&A.

---

## Live Demo

- **Hugging Face Space:** https://huggingface.co/spaces/mehebub-hasan/reviewlens-ai  

---

## Why I Built This

Customer reviews often contain valuable information, but they are usually messy, unstructured, and difficult to analyze manually.

Star ratings alone do not explain:

- what customers actually like,
- what they complain about,
- which issues appear repeatedly,
- which feedback patterns are hidden inside thousands of reviews.

ReviewLens AI was built to turn raw customer feedback into clear, structured, and actionable insights.

---

## Key Features

### Dataset Upload

- Upload any CSV review or feedback dataset.
- Select the text/review column.
- Optionally select the rating column.
- Preview dataset shape and sample rows.

### NLP Preprocessing

- Text cleaning
- Lowercasing
- Stopword removal
- Punctuation removal
- Short-text filtering

### Feature Engineering

- TF-IDF vectorization
- Bigram support
- TruncatedSVD dimensionality reduction

### Clustering

ReviewLens AI includes multiple clustering methods:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

The app supports both:

- **Fast Mode** for smoother and quicker analysis
- **Advanced Mode** for deeper full-pipeline analysis

### Sentiment Analysis

The app includes two types of sentiment analysis:

- Rating-based sentiment analysis
- Deep NLP sentiment analysis using a RoBERTa transformer model

### Interactive Visualizations

- Cluster size chart
- 2D K-Means cluster visualization
- 3D K-Means cluster visualization
- Rating sentiment distribution
- Deep sentiment distribution
- Sentiment comparison across clusters

### Dataset Chatbots

ReviewLens AI includes two chatbot options:

1. **Basic Dataset Chatbot**  
   A rule-based chatbot for quick dataset questions such as:
   - Show cluster size
   - Show top words
   - Show average rating by cluster
   - Show sentiment by cluster
   - Show example reviews

2. **AI Dataset Analyst**  
   An LLM-powered analyst that allows users to ask flexible natural-language questions about the analyzed dataset.

### Export

- Download the fully analyzed CSV dataset.

---

## Example Questions

You can ask the chatbot questions like:

```text
Show cluster size
Show top words
Show average rating by cluster
Which cluster has the lowest rating?
Compare sentiment across clusters
Give me example negative reviews from the worst cluster
What business insight can we get from this dataset?