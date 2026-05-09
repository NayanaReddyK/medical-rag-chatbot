"""
rag_pipeline.py
---------------
Wires everything together:
  Retriever (FAISS or TF-IDF) → Prompt Engine → LLM Client

This is the single entry point called by the Streamlit UI.
"""

import os
import json
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """
    End-to-end Medical RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()

        # Streaming
        for token in pipeline.stream("Can I take metformin with alcohol?"):
            print(token, end="")

        # Blocking
        result = pipeline.query("What causes hypertension?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        model: str | None = None,
        use_hybrid: bool = True,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.model_name      = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.use_hybrid      = use_hybrid
        self.top_k           = top_k or int(os.getenv("TOP_K", "5"))
        self.score_threshold = score_threshold or float(os.getenv("SCORE_THRESHOLD", "0.35"))

        self._retriever  = None
        self._llm_client = None
        self._initialized = False

    def initialize(self, force_rebuild: bool = False) -> None:
        """Load or build the vector store and initialize the LLM client."""
        from embeddings.vector_store import VectorStore, HybridRetriever, get_vector_store
        from retrieval.tfidf_retriever import TFIDFRetriever
        from llm.llm_client import get_llm_client
        from data.preprocessor import build_corpus

        # ── Vector store ──────────────────────────────────────────────────
        vs = get_vector_store()

        if force_rebuild or not vs.is_built():
            print("[RAG] Building vector store from corpus...")
            chunks = build_corpus()
            vs.build(chunks)
        else:
            vs.load()

        # ── TF-IDF (for hybrid) ───────────────────────────────────────────
        tfidf = None
        if self.use_hybrid:
            chunks_path = os.getenv("CHUNKS_PATH", "./embeddings/chunks.json")
            if Path(chunks_path).exists():
                tfidf = TFIDFRetriever(chunks_path)
                tfidf.load()
            else:
                print("[RAG] No chunks.json for TF-IDF; using dense-only.")

        # ── Retriever selection ───────────────────────────────────────────
        if self.use_hybrid and tfidf is not None:
            self._retriever = HybridRetriever(vs, tfidf)
            print("[RAG] Using HybridRetriever (dense + sparse).")
        else:
            self._retriever = vs
            print("[RAG] Using dense-only VectorStore retriever.")

        # ── LLM client ────────────────────────────────────────────────────
        self._llm_client = get_llm_client(self.model_name)
        print(f"[RAG] LLM: {self._llm_client.model_name}")

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def retrieve(
        self,
        query: str,
        source_filter: str | None = None,
        drug_filter: str | None = None,
    ) -> list[dict]:
        """Run retrieval only — useful for debugging."""
        self._ensure_initialized()
        return self._retriever.retrieve(
            query,
            k=self.top_k,
            score_threshold=self.score_threshold,
            source_filter=source_filter,
            drug_filter=drug_filter,
        )

    def query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """
        Blocking RAG query.

        Returns:
            {
                "answer":  str,
                "sources": list[dict],
                "intent":  QueryIntent,
            }
        """
        from llm.prompt_engine import detect_intent, prepare_llm_inputs

        self._ensure_initialized()

        intent = detect_intent(query)

        # Bias retrieval toward DrugBank if drug interaction is detected
        source_filter = "drugbank" if intent.is_drug_interaction and not intent.is_symptom_query else None
        drug_filter   = intent.drugs_mentioned[0] if len(intent.drugs_mentioned) == 1 else None

        chunks = self._retriever.retrieve(
            query,
            k=self.top_k,
            score_threshold=self.score_threshold,
            source_filter=source_filter,
            drug_filter=drug_filter,
        )

        # If drug-biased retrieval returned too few results, widen
        if len(chunks) < 2 and source_filter:
            chunks = self._retriever.retrieve(
                query, k=self.top_k, score_threshold=self.score_threshold
            )

        system, messages = prepare_llm_inputs(query, chunks, chat_history)
        answer = self._llm_client.chat(messages, system_prompt=system)

        return {
            "answer":  answer,
            "sources": chunks,
            "intent":  intent,
        }

    def stream(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Streaming RAG query.
        Yields text tokens. Sources are available only after the stream ends.
        Use stream_with_sources() to also get sources.
        """
        from llm.prompt_engine import detect_intent, prepare_llm_inputs

        self._ensure_initialized()

        intent = detect_intent(query)
        source_filter = "drugbank" if intent.is_drug_interaction and not intent.is_symptom_query else None
        drug_filter   = intent.drugs_mentioned[0] if len(intent.drugs_mentioned) == 1 else None

        chunks = self._retriever.retrieve(
            query,
            k=self.top_k,
            score_threshold=self.score_threshold,
            source_filter=source_filter,
            drug_filter=drug_filter,
        )

        if len(chunks) < 2 and source_filter:
            chunks = self._retriever.retrieve(
                query, k=self.top_k, score_threshold=self.score_threshold
            )

        system, messages = prepare_llm_inputs(query, chunks, chat_history)
        yield from self._llm_client.stream(messages, system_prompt=system)

    def stream_with_sources(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """
        Returns (token_generator, sources_list).
        Sources are resolved before streaming begins.
        """
        from llm.prompt_engine import detect_intent, prepare_llm_inputs

        self._ensure_initialized()

        intent = detect_intent(query)
        source_filter = "drugbank" if intent.is_drug_interaction and not intent.is_symptom_query else None
        drug_filter   = intent.drugs_mentioned[0] if len(intent.drugs_mentioned) == 1 else None

        chunks = self._retriever.retrieve(
            query, k=self.top_k, score_threshold=self.score_threshold,
            source_filter=source_filter, drug_filter=drug_filter,
        )
        if len(chunks) < 2 and source_filter:
            chunks = self._retriever.retrieve(
                query, k=self.top_k, score_threshold=self.score_threshold
            )

        system, messages = prepare_llm_inputs(query, chunks, chat_history)
        gen = self._llm_client.stream(messages, system_prompt=system)
        return gen, chunks, intent


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_pipeline(**kwargs) -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(**kwargs)
    return _pipeline
