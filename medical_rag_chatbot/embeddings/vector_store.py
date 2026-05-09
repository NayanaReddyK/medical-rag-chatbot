"""
vector_store.py
---------------
Builds and queries a FAISS vector store using sentence-transformer embeddings.

Model choices (in order of medical domain quality):
  - "pritamdeka/S-PubMedBert-MS-MARCO"  — biomedical, best quality
  - "all-MiniLM-L6-v2"                   — fast, general purpose (default)
  - "all-mpnet-base-v2"                  — slower, higher quality general
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH    = "./embeddings/faiss.index"
CHUNKS_PATH   = "./embeddings/chunks.json"


class VectorStore:
    """FAISS-backed dense vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        index_path: str = INDEX_PATH,
        chunks_path: str = CHUNKS_PATH,
    ):
        self.model_name  = model_name
        self.index_path  = index_path
        self.chunks_path = chunks_path
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        if self.model is None:
            print(f"[VectorStore] Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    # ── Building ──────────────────────────────────────────────────────────────
    def build(self, chunks: list[dict], batch_size: int = 64) -> None:
        """
        Embed all chunks and create a FAISS index.
        Saves index + chunks to disk automatically.
        """
        self._load_model()
        self.chunks = chunks

        texts = [c["text"] for c in chunks]
        print(f"[VectorStore] Embedding {len(texts)} chunks...")
        t0 = time.time()

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]
            vecs = self.model.encode(
                batch,
                normalize_embeddings=True,   # for cosine similarity via inner product
                show_progress_bar=False,
            )
            embeddings.append(vecs)

        embeddings = np.vstack(embeddings).astype("float32")
        dim = embeddings.shape[1]

        print(f"[VectorStore] Building FAISS index (dim={dim}) ...")
        self.index = faiss.IndexFlatIP(dim)   # Inner Product == cosine after normalization
        self.index.add(embeddings)

        print(f"[VectorStore] Done in {time.time()-t0:.1f}s. "
              f"Indexed {self.index.ntotal} vectors.")
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save(self) -> None:
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "w") as f:
            json.dump(self.chunks, f)
        print(f"[VectorStore] Saved to {self.index_path} + {self.chunks_path}")

    def load(self) -> None:
        """Load index and chunks from disk."""
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run `python embeddings/vector_store.py` to build it first."
            )
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path) as f:
            self.chunks = json.load(f)
        self._load_model()
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {self.index_path}")

    def is_built(self) -> bool:
        return Path(self.index_path).exists() and Path(self.chunks_path).exists()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.35,
        source_filter: str | None = None,
        drug_filter: str | None = None,
    ) -> list[dict]:
        """
        Dense retrieval using cosine similarity.

        Args:
            query:            User query.
            k:                Number of final results.
            score_threshold:  Minimum cosine score to include.
            source_filter:    "medquad" | "drugbank" | None
            drug_filter:      Drug name to bias search toward.

        Returns:
            List of chunk dicts with "score" field added.
        """
        if self.index is None or self.model is None:
            raise RuntimeError("Call .build() or .load() first.")

        # Embed query
        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Search wider than k to allow post-filtering
        search_k = min(k * 4, self.index.ntotal)
        scores, indices = self.index.search(q_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < score_threshold:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)

            # Apply filters
            if source_filter and chunk["source"] != source_filter:
                continue
            if drug_filter:
                drug_lower = drug_filter.lower()
                if (drug_lower not in chunk["category"].lower()
                        and drug_lower not in chunk["text"].lower()):
                    continue

            results.append(chunk)
            if len(results) >= k:
                break

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever — combines TF-IDF sparse + dense scores
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """
    Reciprocal Rank Fusion of TF-IDF + dense results.
    Gives best of both worlds: keyword precision + semantic recall.
    """

    def __init__(self, vector_store: VectorStore, tfidf_retriever=None):
        self.vs = vector_store
        self.tfidf = tfidf_retriever

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.30,
        source_filter: str | None = None,
        drug_filter: str | None = None,
    ) -> list[dict]:
        fetch_k = k * 3

        dense_results = self.vs.retrieve(
            query, k=fetch_k, score_threshold=0.0,
            source_filter=source_filter, drug_filter=drug_filter,
        )

        if self.tfidf is not None:
            sparse_results = self.tfidf.retrieve(
                query, k=fetch_k, score_threshold=0.0,
                source_filter=source_filter, drug_filter=drug_filter,
            )
        else:
            sparse_results = []

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:k]:
            chunk = dict(chunk_map[cid])
            chunk["score"] = rrf_scores[cid]
            results.append(chunk)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessors
# ─────────────────────────────────────────────────────────────────────────────
_vector_store: VectorStore | None = None


def get_vector_store(
    model_name: str = DEFAULT_MODEL,
    index_path: str | None = None,
    chunks_path: str | None = None,
) -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            model_name=model_name,
            index_path=index_path or os.getenv("VECTOR_STORE_PATH", INDEX_PATH),
            chunks_path=chunks_path or os.getenv("CHUNKS_PATH", CHUNKS_PATH),
        )
    return _vector_store


# ─────────────────────────────────────────────────────────────────────────────
# CLI: python embeddings/vector_store.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.preprocessor import build_corpus

    print("Building corpus...")
    chunks = build_corpus()

    vs = VectorStore()
    vs.build(chunks)

    # Quick test
    print("\n--- Test query: 'warfarin aspirin interaction' ---")
    vs.load()
    for r in vs.retrieve("warfarin aspirin interaction", k=3):
        print(f"  [{r['score']:.3f}] {r['source']}/{r['category']}: {r['text'][:100]}")
