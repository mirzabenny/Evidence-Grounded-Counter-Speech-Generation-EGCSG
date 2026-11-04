
import os
import json
import math

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# Load environment configuration
# ============================================================

load_dotenv()

# Input and output paths
CORPUS_FILE = os.getenv("CORPUS_FILE")
FAISS_PATH = os.getenv("FAISS_INDEX_PATH")
TEXT_PATH = os.getenv("TEXT_METADATA_PATH")

# Text chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "40"))

# Embedding and indexing configuration
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8000"))
TRAIN_SAMPLE = int(os.getenv("FAISS_TRAIN_SAMPLE", "50000"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE")
ENCODE_BATCH_SIZE = int(
    os.getenv("ENCODE_BATCH_SIZE", "128")
)

# FAISS configuration
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "100"))

# Validate required paths
required_paths = {
    "CORPUS_FILE": CORPUS_FILE,
    "FAISS_INDEX_PATH": FAISS_PATH,
    "TEXT_METADATA_PATH": TEXT_PATH
}

for key, value in required_paths.items():
    if not value:
        raise ValueError(
            f"{key} is missing from the .env file."
        )

# Validate required model settings
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL is missing from the .env file.")

if not EMBEDDING_DEVICE:
    raise ValueError("EMBEDDING_DEVICE is missing from the .env file.")

if ENCODE_BATCH_SIZE <= 0:
    raise ValueError(
        "ENCODE_BATCH_SIZE must be greater than zero."
    )

# Validate numerical configuration
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero.")

if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
    )

if MIN_CHUNK_WORDS <= 0:
    raise ValueError("MIN_CHUNK_WORDS must be greater than zero.")

if BATCH_SIZE <= 0 or TRAIN_SAMPLE <= 0:
    raise ValueError(
        "EMBEDDING_BATCH_SIZE and FAISS_TRAIN_SAMPLE must be positive."
    )

if EMBEDDING_DIMENSION <= 0 or FAISS_NLIST <= 0:
    raise ValueError(
        "EMBEDDING_DIMENSION and FAISS_NLIST must be positive."
    )


# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text):
    """
    Remove extra spaces and newlines.
    Makes text consistent before embedding.
    """
    return " ".join(text.split())


def chunk_text(text):
    """
    Split long text into overlapping chunks.
    This improves retrieval precision.
    """
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i in range(0, len(words), step):

        chunk_words = words[i:i + CHUNK_SIZE]

        # Skip small chunks (low information)
        if len(chunk_words) < MIN_CHUNK_WORDS:
            continue

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


# ============================================================
# Corpus loading and chunk generation
# ============================================================

def load_corpus():
    """
    Convert each document into multiple chunks.
    Each chunk = Title + Summary + Chunk of Text.
    """
    corpus_chunks = []

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:

        for idx, line in enumerate(tqdm(f, desc="Loading corpus")):

            try:
                item = json.loads(line)

                title = clean_text(item.get("title", ""))
                summary = clean_text(item.get("summary", ""))
                text = clean_text(item.get("text", ""))

                if not text:
                    continue

                # Base context (global information)
                base_context = f"{title}. {summary}".strip()

                # Split text into chunks
                chunks = chunk_text(text)

                for chunk_id, chunk in enumerate(chunks):

                    combined = f"{base_context} {chunk}"

                    corpus_chunks.append({
                        "text": combined,
                        "doc_id": idx,
                        "chunk_id": chunk_id
                    })

            except Exception:
                continue

    return corpus_chunks


# ============================================================
# FAISS index construction
# ============================================================

def build_corpus():

    data = load_corpus()

    print(f"\nTotal chunks created: {len(data)}")

    if not data:
        raise ValueError(
            "No corpus chunks were created. Check the input corpus file."
        )

    texts = [d["text"] for d in data]

    # Initialize embedding model
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE
    )

    # Initialize FAISS index
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

    index = faiss.IndexIVFFlat(
        quantizer,
        EMBEDDING_DIMENSION,
        FAISS_NLIST
    )

    print("Training FAISS index...")

    sample_texts = texts[:TRAIN_SAMPLE]

    sample_embeddings = model.encode(
        sample_texts,
        batch_size=ENCODE_BATCH_SIZE
    )

    sample_embeddings = np.array(
        sample_embeddings
    ).astype("float32")

    if sample_embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Configured embedding dimension is {EMBEDDING_DIMENSION}, "
            f"but the model generated {sample_embeddings.shape[1]}."
        )

    if len(sample_embeddings) < FAISS_NLIST:
        raise ValueError(
            "Insufficient training embeddings for the configured FAISS_NLIST."
        )

    faiss.normalize_L2(sample_embeddings)

    index.train(sample_embeddings)

    print("Adding embeddings in batches...")

    num_batches = math.ceil(len(texts) / BATCH_SIZE)

    for i in tqdm(range(num_batches), desc="Indexing batches"):

        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(texts))

        batch_texts = texts[start:end]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH_SIZE
        )

        batch_embeddings = np.array(
            batch_embeddings
        ).astype("float32")

        faiss.normalize_L2(batch_embeddings)

        index.add(batch_embeddings)

    # Save FAISS index
    faiss.write_index(index, FAISS_PATH)

    print("FAISS index saved.")

    # Save text metadata
    np.save(
        TEXT_PATH,
        np.array(data, dtype=object)
    )

    print("Text metadata saved.")


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    build_corpus()
