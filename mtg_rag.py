"""
MTG Rules RAG System
====================
Retrieval-Augmented Generation for Magic: The Gathering Comprehensive Rules.

Uses TF-IDF for fast local retrieval and a local Ollama model for
natural-language answers with cited rule numbers. Requires Ollama running
locally (`brew install ollama && ollama pull llama3.2`).

Usage:
    python mtg_rag.py setup              # One-time: build TF-IDF index from mtg_rules.txt
    python mtg_rag.py query "question"   # Ask a single question
    python mtg_rag.py                    # Interactive Q&A mode

Requirements:
    pip install -r requirements.txt
    # No API key needed — model runs locally via Ollama
"""

import os
import re
import sys
import pickle

RULES_FILE = "mtg_rules.txt"
INDEX_FILE = "mtg_index.pkl"
TOP_K = 6
MIN_CHUNK_CHARS = 80

OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are an expert Magic: The Gathering rules judge. "
    "Answer the player's question using ONLY the rules provided below. "
    "Cite the specific rule numbers (e.g. 702.7b) that support your answer. "
    "Be concise and precise."
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def load_and_chunk_rules(path: str) -> list[dict]:
    """
    Split mtg_rules.txt into semantically coherent chunks.

    Each chunk covers one numbered rule group: the parent rule heading plus
    all its lettered sub-rules.
    Example: rule 702.7 (First Strike) becomes a single chunk containing
    702.7., 702.7a, 702.7b, etc.

    Returns a list of dicts: {"id": str, "text": str}
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # Pattern that marks the start of a new parent rule:
    #   "702.7. First Strike"  or  "100.1. These Magic rules..."
    # Sub-rules use letters with no trailing period: 702.7a, 702.7b
    parent_rule_pat = re.compile(r'(?m)^(\d{3,4}\.\d+\.)\s')

    chunks = []
    matches = list(parent_rule_pat.finditer(text))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue

        rule_id = m.group(1)
        chunks.append({"id": rule_id, "text": chunk_text})

    print(f"[setup] Loaded {len(chunks)} rule chunks from {path}")
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF index (no HuggingFace downloads required)
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict]):
    """Fit a TF-IDF vectoriser over all chunks and return (vectoriser, matrix)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for better phrase matching
        sublinear_tf=True,    # dampen term frequency
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def setup():
    """Build and persist the TF-IDF index from mtg_rules.txt."""
    if not os.path.exists(RULES_FILE):
        sys.exit(
            f"[error] '{RULES_FILE}' not found.\n"
            "  Make sure the rules text file is in the same directory."
        )

    chunks = load_and_chunk_rules(RULES_FILE)
    print("[setup] Building TF-IDF index...")
    vectorizer, matrix = build_index(chunks)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, f)

    print(f"[setup] Index saved to '{INDEX_FILE}' ({os.path.getsize(INDEX_FILE)//1024} KB)")


def load_index():
    """Load the persisted index. Raises RuntimeError if not built yet."""
    if not os.path.exists(INDEX_FILE):
        raise RuntimeError(
            f"Index file '{INDEX_FILE}' not found. Run setup first:\n"
            "  python mtg_rag.py setup"
        )
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str, index: dict, top_k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant rule chunks for the question."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]

    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def query(question: str) -> str:
    """Retrieve relevant rule chunks and generate an answer with the local LLM."""
    import ollama

    index = load_index()
    hits = retrieve(question, index)

    if not hits:
        context = "(No closely matching rules found.)"
    else:
        parts = [f"[Rule {c['id']}]\n{c['text']}" for c in hits]
        context = "\n\n---\n\n".join(parts)

    prompt = (
        f"Relevant Rules:\n\n{context}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer based on the rules above, citing rule numbers."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if args and args[0] == "setup":
        setup()
        return

    if args and args[0] == "query":
        if len(args) < 2:
            print('Usage: python mtg_rag.py query "your question"')
            sys.exit(1)
        question = " ".join(args[1:])
        print(f"\nQuestion: {question}\n")
        try:
            answer = query(question)
        except RuntimeError as e:
            sys.exit(f"[error] {e}")
        print(f"Answer:\n{answer}\n")
        return

    # Wrap load_index errors as friendly CLI messages
    try:
        load_index()  # validate index exists before entering interactive mode
    except RuntimeError as e:
        sys.exit(f"[error] {e}")

    # Interactive mode
    print("MTG Rules Q&A  (type 'quit' to exit)")
    print("=" * 50)
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        answer = query(question)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
