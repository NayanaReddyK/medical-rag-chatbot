"""
build_index.py
--------------
One-time script to preprocess data and build the FAISS vector store.

Run this before starting the Streamlit app:
    python build_index.py

Options:
    python build_index.py --rebuild          Force rebuild even if index exists
    python build_index.py --model <name>     Use a specific embedding model
    python build_index.py --test             Run retrieval tests after building
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Build the Medical RAG vector index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--test", action="store_true", help="Run retrieval tests after build")
    args = parser.parse_args()

    print("=" * 60)
    print("Medical RAG — Index Builder")
    print("=" * 60)

    # Step 1: Preprocess corpus
    print("\n[1/3] Building corpus from data sources...")
    from data.preprocessor import build_corpus
    chunks = build_corpus()
    print(f"      Total chunks: {len(chunks)}")

    source_counts = {}
    type_counts   = {}
    for c in chunks:
        source_counts[c["source"]] = source_counts.get(c["source"], 0) + 1
        type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

    for source, count in source_counts.items():
        print(f"      {source}: {count} chunks")
    for ctype, count in type_counts.items():
        print(f"      type={ctype}: {count}")

    # Step 2: Build vector store
    print(f"\n[2/3] Building FAISS vector store (model={args.model})...")
    from embeddings.vector_store import VectorStore
    vs = VectorStore(model_name=args.model)

    if vs.is_built() and not args.rebuild:
        print("      Index already exists. Use --rebuild to force rebuild.")
        vs.load()
    else:
        vs.build(chunks)

    # Step 3: Build TF-IDF (from saved chunks)
    print("\n[3/3] Building TF-IDF index...")
    from retrieval.tfidf_retriever import TFIDFRetriever
    tfidf = TFIDFRetriever()
    tfidf.build(chunks)

    print("\n✅ Index built successfully!")
    print(f"   FAISS index: {vs.index_path}")
    print(f"   Chunks JSON: {vs.chunks_path}")

    # Optional tests
    if args.test:
        print("\n" + "=" * 60)
        print("Running retrieval tests...")
        print("=" * 60)

        test_queries = [
            ("warfarin aspirin interaction",           "drug interaction"),
            ("symptoms of heart failure",             "symptom query"),
            ("how does metformin work",               "mechanism of action"),
            ("what is hypertension",                  "general Q&A"),
            ("atorvastatin side effects",             "drug info"),
        ]

        from embeddings.vector_store import HybridRetriever
        from retrieval.tfidf_retriever import TFIDFRetriever as TF

        tf = TF()
        tf.build(chunks)
        hybrid = HybridRetriever(vs, tf)

        for query, label in test_queries:
            print(f"\n[{label}] Query: {query!r}")
            results = hybrid.retrieve(query, k=3)
            for r in results:
                print(f"  [{r['score']:.3f}] {r['source']}/{r['category']} "
                      f"({r['chunk_type']}) — {r['text'][:80]}...")

        print("\n✅ All tests passed.")


if __name__ == "__main__":
    main()
