
import os
import json
import requests
import numpy as np
import pandas as pd
import faiss
import torch

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu


# ============================================================
# LOAD ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


def get_required_env(name):
    """Read a required environment variable."""
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"{name} is missing from the .env file."
        )

    return value


def get_int_env(name):
    """Read an integer environment variable."""
    value = get_required_env(name)

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"{name} must contain an integer value."
        )


# ------------------------------------------------------------
# Input and output paths
# ------------------------------------------------------------

FAISS_PATH = get_required_env("FAISS_INDEX_PATH")
TEXT_PATH = get_required_env("TEXT_METADATA_PATH")
DATASET_PATH = get_required_env("GENERATION_DATASET_PATH")
OUTPUT_JSONL = get_required_env("GENERATION_OUTPUT_PATH")


# ------------------------------------------------------------
# Ollama configuration
# ------------------------------------------------------------

OLLAMA_URL = get_required_env("OLLAMA_URL")
MODEL = get_required_env("OLLAMA_MODEL")
OLLAMA_TIMEOUT = get_int_env("OLLAMA_TIMEOUT")


# ------------------------------------------------------------
# Tavily configuration
# ------------------------------------------------------------

TAVILY_API_KEY = get_required_env("TAVILY_API_KEY")
TAVILY_URL = get_required_env("TAVILY_URL")
TAVILY_SEARCH_DEPTH = get_required_env("TAVILY_SEARCH_DEPTH")
TAVILY_MAX_RESULTS = get_int_env("TAVILY_MAX_RESULTS")
TAVILY_TOKEN_LIMIT = get_int_env("TAVILY_TOKEN_LIMIT")
TAVILY_TIMEOUT = get_int_env("TAVILY_TIMEOUT")


# ------------------------------------------------------------
# Google Perspective API configuration
# ------------------------------------------------------------

PERSPECTIVE_API_KEY = get_required_env("PERSPECTIVE_API_KEY")
PERSPECTIVE_URL = get_required_env("PERSPECTIVE_URL")
PERSPECTIVE_TIMEOUT = get_int_env("PERSPECTIVE_TIMEOUT")
PERSPECTIVE_LANGUAGE = get_required_env("PERSPECTIVE_LANGUAGE")
PERSPECTIVE_ATTRIBUTE = get_required_env("PERSPECTIVE_ATTRIBUTE")


# ------------------------------------------------------------
# Embedding configuration
# ------------------------------------------------------------

EMBEDDING_MODEL = get_required_env("EMBEDDING_MODEL")
EMBEDDING_DEVICE = get_required_env("EMBEDDING_DEVICE")


# ------------------------------------------------------------
# FAISS retrieval configuration
# ------------------------------------------------------------

FAISS_NPROBE = get_int_env("FAISS_NPROBE")
RETRIEVAL_TOP_K = get_int_env("RETRIEVAL_TOP_K")
NUM_SEARCH_QUERIES = get_int_env("NUM_SEARCH_QUERIES")


# ------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------

ROUGE_METRIC = get_required_env("ROUGE_METRIC")
BERTSCORE_LANGUAGE = get_required_env("BERTSCORE_LANGUAGE")
BLEU_TOKENIZATION = get_required_env("BLEU_TOKENIZATION")


# Validate configuration

if RETRIEVAL_TOP_K <= 0 or NUM_SEARCH_QUERIES <= 0:
    raise ValueError(
        "RETRIEVAL_TOP_K and NUM_SEARCH_QUERIES must be positive."
    )

if FAISS_NPROBE <= 0:
    raise ValueError("FAISS_NPROBE must be positive.")


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

if EMBEDDING_DEVICE.lower() == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA was selected in .env, but CUDA is not available."
    )

DEVICE = torch.device(EMBEDDING_DEVICE)

print("Using device:", DEVICE)


# ============================================================
# LOAD FAISS INDEX AND TEXT METADATA
# ============================================================

index = faiss.read_index(FAISS_PATH)

texts = np.load(
    TEXT_PATH,
    allow_pickle=True
)

embedder = SentenceTransformer(
    EMBEDDING_MODEL,
    device=DEVICE.type
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    try:
        text = text.encode(
            "latin1"
        ).decode(
            "utf-8"
        )
    except Exception:
        pass

    text = " ".join(text.split())

    return text.strip()


def clean_retrieved_text(text):

    text = normalize_text(text)

    if len(text) > 0:

        if text[-1] not in ".!?\"'”":

            last_period = text.rfind(".")

            if last_period != -1:
                text = text[:last_period + 1]

    return text.strip()


# ============================================================
# OLLAMA GENERATION
# ============================================================

def call_ollama(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=OLLAMA_TIMEOUT
    )

    response.raise_for_status()

    full_response = ""

    for line in response.iter_lines():

        if line:

            try:
                data = json.loads(
                    line.decode(
                        "utf-8",
                        errors="ignore"
                    )
                )

                full_response += data.get(
                    "response",
                    ""
                )

            except Exception:
                pass

    return full_response.strip()


def clean_output_text(text):

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text.strip()


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def generate_queries(hate_speech):

    prompt = f"""
Generate exactly {NUM_SEARCH_QUERIES} factual search queries
to verify claims in:

{hate_speech}

Format:
1. ...
2. ...
"""

    raw = call_ollama(prompt)

    queries = []

    for line in raw.splitlines():

        line = line.strip()

        if line and line[0].isdigit() and "." in line:

            q = line.split(".", 1)[1].strip()

            q = q.replace(
                "Search Query:",
                ""
            ).strip()

            q = q.split(" - ")[0].strip()

            q = q.replace('\\"', '')
            q = q.replace('"', '')
            q = q.replace("'", '')

            q = normalize_text(q)

            if q:
                queries.append(q)

    return queries[:NUM_SEARCH_QUERIES]


# ============================================================
# STATIC KNOWLEDGE BASE RETRIEVAL
# ============================================================

def search_corpus(query, k=None):

    if k is None:
        k = RETRIEVAL_TOP_K

    q_emb = embedder.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(q_emb)

    index.nprobe = FAISS_NPROBE

    distances, indices = index.search(
        q_emb,
        k
    )

    results = []

    for score, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx == -1:
            continue

        doc = texts[idx]

        results.append({

            "text": clean_retrieved_text(
                doc.get(
                    "text",
                    ""
                )
            ),

            "similarity_score": float(
                f"{score:.4f}"
            )

        })

    return results


# ============================================================
# TAVILY WEB SEARCH
# ============================================================

def tavily_search(query, k=None):

    if k is None:
        k = TAVILY_MAX_RESULTS

    payload = {

        "api_key": TAVILY_API_KEY,

        "query": query,

        "search_depth": TAVILY_SEARCH_DEPTH,

        "max_results": k,

        "n_tokens": TAVILY_TOKEN_LIMIT
    }

    try:

        response = requests.post(
            TAVILY_URL,
            json=payload,
            timeout=TAVILY_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        results = []

        if "results" in data:

            for r in data["results"]:

                text = clean_retrieved_text(
                    r.get(
                        "content",
                        ""
                    )
                )

                if text:

                    results.append({
                        "text": text
                    })

        return results

    except Exception as e:

        print("Tavily Error:", e)

        return []


# ============================================================
# DOCUMENT RELEVANCE GRADING
# ============================================================

def grade(document, query, hate_speech):

    prompt = f"""
Determine whether the document is relevant
for countering the hate speech.

Query:

{query}

Hate Speech:

{hate_speech}

Document:

{document}

Reject documents that:
- reinforce stereotypes
- promote hate
- contain conspiracy theories
- support harmful claims

Accept only documents that:
- provide factual correction
- provide historical clarification
- debunk stereotypes
- provide constructive evidence

Answer ONLY:

relevant

or

irrelevant
"""

    response = call_ollama(
        prompt
    ).lower().strip()

    return response.startswith(
        "relevant"
    )


# ============================================================
# EVIDENCE AGGREGATION
# ============================================================

def aggregate_evidence(
    documents,
    hate_speech
):

    combined_docs = "\n\n".join(
        documents
    )

    prompt = f"""
You are given multiple retrieved documents
related to hate speech.

Create a concise factual evidence summary.

Requirements:
- 250-300 words maximum
- Remove repetition
- Keep strongest factual evidence
- Maintain factual accuracy
- Focus only on evidence useful for countering hate speech

Hate Speech:

{hate_speech}

Documents:

{combined_docs}

Concise Evidence Summary:
"""

    return normalize_text(
        clean_output_text(
            call_ollama(prompt)
        )
    )


# ============================================================
# COUNTER-SPEECH GENERATION
# ============================================================

def generate_cs(
    summary,
    hate_speech
):

    prompt = f"""
Generate factual counter speech
in 3-4 respectful sentences.

Use ONLY the evidence below.

Evidence:

{summary}

Hate Speech:

{hate_speech}

Counter Speech:
"""

    return normalize_text(
        clean_output_text(
            call_ollama(prompt)
        )
    )


# ============================================================
# GENERATED COUNTER-SPEECH CLEANING
# ============================================================

def clean_generated_cs(text):

    prompt = f"""
Clean the following counter-speech text by removing:

- malformed citation tokens
- broken formatting symbols
- HTML/XML remnants
- retrieval artifacts
- repeated special characters
- meaningless noisy tokens
- source references
- bracketed citations
- incomplete URLs

Do NOT:
- change factual meaning
- rewrite sentences
- summarize
- add new information

Return ONLY the cleaned paragraph.

Text:

{text}
"""

    cleaned = call_ollama(prompt)

    return normalize_text(
        clean_output_text(
            cleaned
        )
    )


# ============================================================
# TOXICITY EVALUATION
# ============================================================

def get_toxicity_score(text):

    payload = {

        "comment": {
            "text": text
        },

        "languages": [
            PERSPECTIVE_LANGUAGE
        ],

        "requestedAttributes": {
            PERSPECTIVE_ATTRIBUTE: {}
        }
    }

    try:

        response = requests.post(
            PERSPECTIVE_URL,
            params={
                "key": PERSPECTIVE_API_KEY
            },
            json=payload,
            timeout=PERSPECTIVE_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "attributeScores"
        ][
            PERSPECTIVE_ATTRIBUTE
        ][
            "summaryScore"
        ][
            "value"
        ]

    except Exception as e:

        print(
            "Perspective API Error:",
            e
        )

        return None


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_scores(
    reference,
    generated
):

    scorer = rouge_scorer.RougeScorer(
        [ROUGE_METRIC],
        use_stemmer=True
    )

    rouge2 = scorer.score(
        reference,
        generated
    )

    bleu2 = sacrebleu.corpus_bleu(
        [generated],
        [[reference]],
        tokenize=BLEU_TOKENIZATION
    ).score

    P, R, F1 = bert_score(
        [generated],
        [reference],
        lang=BERTSCORE_LANGUAGE
    )

    toxicity = get_toxicity_score(
        generated
    )

    return {

        "rouge2": str(rouge2),

        "bleu2": bleu2,

        "bert_f1": float(F1[0]),

        "toxicity": toxicity
    }


# ============================================================
# PROCESS SINGLE HATE-SPEECH INSTANCE
# ============================================================

def process_row(hate_speech):

    hate_speech = normalize_text(
        hate_speech
    )

    queries = generate_queries(
        hate_speech
    )

    # Fallback if fewer than the required queries are generated
    while len(queries) < NUM_SEARCH_QUERIES:
        queries.append(hate_speech)

    queries = queries[:NUM_SEARCH_QUERIES]

    # Retrieve evidence from FAISS and Tavily
    faiss_results = []
    tavily_results = []

    for query in queries:

        faiss_results.append(
            search_corpus(
                query,
                k=RETRIEVAL_TOP_K
            )
        )

        tavily_results.append(
            tavily_search(
                query,
                k=TAVILY_MAX_RESULTS
            )
        )

    all_docs = []

    for query, results in zip(queries, faiss_results):

        for d in results:

            all_docs.append({

                "source": "faiss",

                "query": query,

                "text": d["text"]
            })

    for query, results in zip(queries, tavily_results):

        for d in results:

            all_docs.append({

                "source": "tavily",

                "query": query,

                "text": d["text"]
            })

    # Grade retrieved documents
    relevant_docs = []

    for d in all_docs:

        is_relevant = grade(
            d["text"],
            d["query"],
            hate_speech
        )

        if is_relevant:
            relevant_docs.append(d)

    # Fallback to first Tavily result if no document is relevant
    if len(relevant_docs) == 0:

        if tavily_results and len(tavily_results[0]) > 0:

            relevant_docs.append({

                "source": "tavily",

                "query": queries[0],

                "text": tavily_results[0][0]["text"]
            })

    doc_texts = [
        d["text"]
        for d in relevant_docs
    ]

    # Evidence summarization
    final_summary = aggregate_evidence(
        doc_texts,
        hate_speech
    )

    # Counter-speech generation
    counter_speech = generate_cs(
        final_summary,
        hate_speech
    )

    counter_speech = clean_generated_cs(
        counter_speech
    )

    # Evaluation
    scores = compute_scores(
        final_summary,
        counter_speech
    )

    # Prepare output
    retrieval_output = {}

    for i, query in enumerate(queries):

        retrieval_output[f"query_{i + 1}"] = {

            "query": query,

            "faiss_results": faiss_results[i],

            "tavily_results": tavily_results[i]
        }

    output = {

        "hate_speech": hate_speech,

        "retrieval_results": retrieval_output,

        "num_relevant_docs": len(relevant_docs),

        "summary": final_summary,

        "generated_counter_speech": counter_speech,

        "scores": scores
    }

    return output


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(df):

    output_directory = os.path.dirname(
        os.path.abspath(OUTPUT_JSONL)
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            hate_speech = normalize_text(
                row["hate_speech"]
            )

            result = process_row(
                hate_speech
            )

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2
                )
                + "\n\n"
            )

    print(
        "\nResults saved to:"
    )

    print(OUTPUT_JSONL)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(
        DATASET_PATH,
        encoding="utf-8-sig",
        encoding_errors="ignore",
        engine="python"
    )

    df["hate_speech"] = (
        df["hate_speech"]
        .astype(str)
        .apply(normalize_text)
    )

    process_dataset(df)
