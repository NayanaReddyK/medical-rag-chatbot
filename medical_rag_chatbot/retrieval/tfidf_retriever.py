"""
tfidf_retriever.py
------------------
TF-IDF baseline retriever.
Works without GPU or embedding models — useful for fast testing
and as a fallback when the vector store isn't built yet.
"""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Medical stopwords to add on top of sklearn's default English list
_MEDICAL_STOPWORDS = {
    "patient", "patients", "doctor", "doctors", "treatment", "treatments",
    "disease", "diseases", "condition", "conditions", "symptom", "symptoms",
    "used", "using", "cause", "causes", "caused", "may", "can", "also",
    "associated", "include", "including", "however", "although", "without",
    "often", "common", "commonly",
}


class TFIDFRetriever:
    """Sparse BM25-style retriever using TF-IDF."""

    def __init__(self, chunks_path: str = "./embeddings/chunks.json"):
        self.chunks_path = chunks_path
        self.chunks: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, chunks: list[dict]) -> None:
        """Build the TF-IDF index from a chunk list."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            stop_words=list(_MEDICAL_STOPWORDS),
            ngram_range=(1, 2),
            max_df=0.90,
            min_df=1,
            sublinear_tf=True,         # log(1 + tf) — reduces impact of very frequent terms
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"[TF-IDF] Index built: {len(chunks)} chunks, "
              f"{self.matrix.shape[1]} features.")

    def load(self) -> None:
        """Load chunks from disk and rebuild index."""
        path = Path(self.chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")
        with open(path) as f:
            chunks = json.load(f)
        self.build(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.05,
        source_filter: str | None = None,
        drug_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query:            User query string.
            k:                Number of results to return.
            score_threshold:  Minimum cosine similarity to include.
            source_filter:    "medquad" | "drugbank" | None (all)
            drug_filter:      Drug name to bias toward (None = no filter)

        Returns:
            List of chunk dicts with an added "score" field.
        """
        if self.vectorizer is None:
            raise RuntimeError("Call .build() or .load() first.")

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        # Filter by source if requested
        candidates = list(enumerate(sims))
        if source_filter:
            candidates = [
                (i, s) for i, s in candidates
                if self.chunks[i]["source"] == source_filter
            ]
        if drug_filter:
            drug_lower = drug_filter.lower()
            candidates = [
                (i, s) for i, s in candidates
                if drug_lower in self.chunks[i]["category"].lower()
                or drug_lower in self.chunks[i]["text"].lower()
            ]

        # Sort and threshold
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in candidates[:k]:
            if score < score_threshold:
                break
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results


# Singleton for easy import
_retriever: TFIDFRetriever | None = None


def get_tfidf_retriever(chunks_path: str = "./embeddings/chunks.json") -> TFIDFRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TFIDFRetriever(chunks_path)
    return _retriever


if __name__ == "__main__":
    from data.preprocessor import build_corpus
    r = TFIDFRetriever()
    r.build(build_corpus())
    results = r.retrieve("metformin alcohol interaction", k=5)
    for res in results:
        print(f"[{res['score']:.3f}] {res['source']} / {res['category']}")
        print(res["text"][:120], "\n")
