import json
import requests
import numpy as np
import pandas as pd
import faiss
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import sacrebleu




FAISS_PATH = r"C:\Users\ACER\fact_cn\corpus_improved.index"

TEXT_PATH = r"C:\Users\ACER\fact_cn\texts_improved.npy"

DATASET_PATH = (
    r"C:\Users\ACER\fact_cn\Original dataset"
    r"\multitarget_KN_grounded_CN_unique.csv"
)

OUTPUT_JSONL = (
    r"C:\Users\ACER\fact_cn\results"
    r"\KN_results.jsonl"
)




OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "mistral:7b"

TAVILY_API_KEY = "TAVILY_API_KEY"

TAVILY_URL = "https://api.tavily.com/search"

PERSPECTIVE_API_KEY = "AIzaSyAHfRzBoCvk5yeqc3x-fi3qILangbLpShs"

PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    f"?key={PERSPECTIVE_API_KEY}"
)




DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)





index = faiss.read_index(FAISS_PATH)

texts = np.load(
    TEXT_PATH,
    allow_pickle=True
)



embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE.type
)



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

    except:
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




def call_ollama(prompt):

    payload = {

        "model": MODEL,

        "prompt": prompt
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True
    )

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

            except:
                pass

    return full_response.strip()




def clean_output_text(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()


def generate_queries(hate_speech):

    prompt = f"""
Generate exactly 2 factual search queries
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

        if line.startswith(("1.", "2.")):

            q = line.split(".", 1)[1].strip()

            q = q.replace(
                "Search Query:",
                ""
            ).strip()

            q = q.split(" - ")[0].strip()

            q = q.replace('\\"', '')

            q = q.replace('"', '')

            q = q.replace("'", "")

            q = normalize_text(q)

            queries.append(q)

    return queries[:2]




def search_corpus(query, k=2):

    q_emb = embedder.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(q_emb)

    index.nprobe = 10

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

            "text":
                clean_retrieved_text(
                    doc.get(
                        "text",
                        ""
                    )
                ),

            "similarity_score":
                float(f"{score:.4f}")
        })

    return results


def tavily_search(query, k=2):

    payload = {

        "api_key": TAVILY_API_KEY,

        "query": query,

        "search_depth": "advanced",

        "max_results": k,

        "n_tokens": 2048
    }

    try:

        response = requests.post(
            TAVILY_URL,
            json=payload,
            timeout=25
        )

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

def get_toxicity_score(text):

    payload = {

        "comment": {
            "text": text
        },

        "languages": ["en"],

        "requestedAttributes": {
            "TOXICITY": {}
        }
    }

    try:

        response = requests.post(
            PERSPECTIVE_URL,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "attributeScores"
        ][
            "TOXICITY"
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



def compute_scores(
    reference,
    generated
):

    scorer = rouge_scorer.RougeScorer(
        ["rouge2"],
        use_stemmer=True
    )

    rouge2 = scorer.score(
        reference,
        generated
    )

    bleu2 = sacrebleu.corpus_bleu(
        [generated],
        [[reference]]
    ).score

    P, R, F1 = bert_score(
        [generated],
        [reference],
        lang="en"
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


def process_row(hate_speech):

    hate_speech = normalize_text(
        hate_speech
    )


    queries = generate_queries(
        hate_speech
    )

    if len(queries) < 2:

        queries = (
            queries +
            [hate_speech]
        )

    q1, q2 = queries[:2]


    faiss_q1 = search_corpus(
        q1,
        k=2
    )

    faiss_q2 = search_corpus(
        q2,
        k=2
    )

    tav_q1 = tavily_search(
        q1,
        k=2
    )

    tav_q2 = tavily_search(
        q2,
        k=2
    )


    all_docs = []

    for d in faiss_q1:

        all_docs.append({

            "source": "faiss",

            "query": q1,

            "text": d["text"]
        })

    for d in faiss_q2:

        all_docs.append({

            "source": "faiss",

            "query": q2,

            "text": d["text"]
        })

    for d in tav_q1:

        all_docs.append({

            "source": "tavily",

            "query": q1,

            "text": d["text"]
        })

    for d in tav_q2:

        all_docs.append({

            "source": "tavily",

            "query": q2,

            "text": d["text"]
        })


    relevant_docs = []

    for d in all_docs:

        is_relevant = grade(
            d["text"],
            d["query"],
            hate_speech
        )

        if is_relevant:

            relevant_docs.append(d)


    if len(relevant_docs) == 0:

        if len(tav_q1) > 0:

            relevant_docs.append({

                "source": "tavily",

                "query": q1,

                "text": tav_q1[0]["text"]
            })


    doc_texts = [

        d["text"]

        for d in relevant_docs
    ]

    final_summary = aggregate_evidence(
        doc_texts,
        hate_speech
    )


    counter_speech = generate_cs(
        final_summary,
        hate_speech
    )


    counter_speech = clean_generated_cs(
        counter_speech
    )


    scores = compute_scores(
        final_summary,
        counter_speech
    )


    output = {

        "hate_speech": hate_speech,

        "retrieval_results": {

            "query_1": {

                "query": q1,

                "faiss_results": faiss_q1,

                "tavily_results": tav_q1
            },

            "query_2": {

                "query": q2,

                "faiss_results": faiss_q2,

                "tavily_results": tav_q2
            }
        },

        "num_relevant_docs":
            len(relevant_docs),

        "summary":
            final_summary,

        "generated_counter_speech":
            counter_speech,

        "scores":
            scores
    }

    return output



def process_dataset(df):

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