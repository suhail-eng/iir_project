import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "papers.csv")


def ensure_dataset(path: str) -> None:
    """Create a sample dataset if it does not exist or is too small."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            if len(existing) >= 100 and {"title", "abstract", "author", "year", "category"}.issubset(existing.columns):
                return
        except Exception:
            pass

    categories = [
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science",
        "Computer Vision",
        "Natural Language Processing",
        "Information Retrieval",
        "Cybersecurity",
        "Cloud Computing",
        "Robotics",
        "Bioinformatics",
    ]

    topic_terms = {
        "Artificial Intelligence": "intelligent agents reasoning planning knowledge representation",
        "Machine Learning": "supervised unsupervised models optimization training data",
        "Data Science": "data analytics statistics visualization pipeline insights",
        "Computer Vision": "image recognition object detection feature extraction",
        "Natural Language Processing": "language models text mining semantic understanding",
        "Information Retrieval": "query expansion indexing ranking retrieval feedback",
        "Cybersecurity": "network security anomaly detection threat intelligence",
        "Cloud Computing": "distributed systems virtualization scalable infrastructure",
        "Robotics": "autonomous navigation control sensors path planning",
        "Bioinformatics": "genomic data sequence analysis biological networks",
    }

    rows: List[Dict[str, str]] = []
    for idx in range(1, 121):
        category = categories[(idx - 1) % len(categories)]
        year = 2010 + (idx % 15)
        author = f"Dr. Author {idx}"
        title = f"Advanced Study {idx}: {category} for Modern Research"
        abstract = (
            f"This paper investigates {category.lower()} techniques with focus on "
            f"{topic_terms[category]}. The study presents experimental evaluation, "
            f"benchmark comparisons, and implications for AI data learning workflows "
            f"in academic and industrial environments."
        )
        rows.append(
            {
                "title": title,
                "abstract": abstract,
                "author": author,
                "year": year,
                "category": category,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def load_dataset() -> pd.DataFrame:
    ensure_dataset(DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    required_cols = {"title", "abstract", "author", "year", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    return df.fillna("")


def preprocess_text(text: str) -> List[str]:
    text = str(text).lower()
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    return [tok for tok in tokens if tok not in ENGLISH_STOP_WORDS]


def compute_precision_at_k(top_docs: List[Dict], k_max: int = 10) -> List[Dict]:
    precision_rows = []
    relevant_count = 0
    for i, doc in enumerate(top_docs[:k_max], start=1):
        if doc["is_relevant"]:
            relevant_count += 1
        precision_rows.append(
            {
                "k": i,
                "doc_title": doc["title"],
                "is_relevant": doc["is_relevant"],
                "precision": round(relevant_count / i, 4),
            }
        )
    return precision_rows


def run_search(query: str, df: pd.DataFrame) -> Dict:
    query_tokens = preprocess_text(query)
    documents = (df["title"] + " " + df["abstract"]).tolist()

    if not query_tokens:
        return {
            "query_tokens": [],
            "results": [],
            "tf_table": [],
            "tfidf_table": [],
            "cosine_table": [],
            "precision_table": [],
            "chart_payload": {},
        }

    # Manual TF counts for query terms across all documents.
    doc_tokens_list = [preprocess_text(doc) for doc in documents]
    tf_table = []
    for idx, tokens in enumerate(doc_tokens_list):
        counts = {term: tokens.count(term) for term in query_tokens}
        tf_table.append({"doc_index": idx, "title": df.iloc[idx]["title"], **counts})

    # TF-IDF vectorization and cosine similarity.
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([" ".join(query_tokens)])
    similarities = cosine_similarity(query_vector, doc_matrix).flatten()

    feature_names = vectorizer.get_feature_names_out()
    vocab = {term: i for i, term in enumerate(feature_names)}

    tfidf_table = []
    for i in range(len(documents)):
        row = {"doc_index": i, "title": df.iloc[i]["title"]}
        for term in query_tokens:
            col_idx = vocab.get(term)
            row[term] = round(float(doc_matrix[i, col_idx]), 6) if col_idx is not None else 0.0
        tfidf_table.append(row)

    all_results = []
    for i, sim in enumerate(similarities):
        term_tf = {term: tf_table[i][term] for term in query_tokens}
        term_tfidf = {term: tfidf_table[i][term] for term in query_tokens}

        # Relevance heuristic: any query term exists in doc.
        is_relevant = any(v > 0 for v in term_tf.values())
        all_results.append(
            {
                "doc_index": i,
                "rank": 0,
                "title": df.iloc[i]["title"],
                "author": df.iloc[i]["author"],
                "year": int(df.iloc[i]["year"]),
                "category": df.iloc[i]["category"],
                "abstract": df.iloc[i]["abstract"],
                "cosine_similarity": round(float(sim), 6),
                "tf_values": term_tf,
                "tfidf_values": term_tfidf,
                "is_relevant": is_relevant,
            }
        )

    ranked = sorted(all_results, key=lambda x: x["cosine_similarity"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    top_n = ranked[:10]
    precision_table = compute_precision_at_k(top_n, k_max=10)

    top_indices = {item["doc_index"] for item in top_n}
    tf_table_top = [row for row in tf_table if row["doc_index"] in top_indices]
    tfidf_table_top = [row for row in tfidf_table if row["doc_index"] in top_indices]

    cosine_table = [
        {"title": item["title"], "cosine_similarity": item["cosine_similarity"], "rank": item["rank"]}
        for item in top_n
    ]

    chart_payload = {
        "labels": [f"R{item['rank']}" for item in top_n],
        "titles": [item["title"] for item in top_n],
        "cosine_values": [item["cosine_similarity"] for item in top_n],
        "tf_sums": [int(sum(item["tf_values"].values())) for item in top_n],
        "tfidf_sums": [round(sum(item["tfidf_values"].values()), 6) for item in top_n],
        "precision_k": [row["k"] for row in precision_table],
        "precision_values": [row["precision"] for row in precision_table],
    }

    return {
        "query_tokens": query_tokens,
        "results": top_n,
        "tf_table": tf_table_top,
        "tfidf_table": tfidf_table_top,
        "cosine_table": cosine_table,
        "precision_table": precision_table,
        "chart_payload": chart_payload,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    df = load_dataset()
    query = ""
    search_data = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            search_data = run_search(query, df)

    return render_template("index.html", query=query, search_data=search_data)


@app.route("/dataset")
def dataset():
    df = load_dataset()
    papers = df.to_dict(orient="records")
    return render_template("dataset.html", papers=papers)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
