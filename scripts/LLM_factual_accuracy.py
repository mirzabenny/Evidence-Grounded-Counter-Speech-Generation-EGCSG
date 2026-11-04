
import json
import os
import re
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv()


def get_required_env(name):
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Missing environment variable: {name}. "
            "Please configure it in your .env file."
        )

    return value


def get_int_env(name):
    value = get_required_env(name)

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be an integer."
        )


# File paths
GENERATED_JSONL = get_required_env(
    "FACTUAL_ACCURACY_INPUT_JSONL"
)

OUTPUT_JSONL = get_required_env(
    "FACTUAL_ACCURACY_OUTPUT_JSONL"
)

# Ollama configuration
OLLAMA_URL = get_required_env("OLLAMA_URL")
MODEL = get_required_env("OLLAMA_MODEL")
OLLAMA_TIMEOUT = get_int_env("OLLAMA_TIMEOUT")


# Validate input file
if not os.path.isfile(GENERATED_JSONL):
    raise FileNotFoundError(
        f"Input file not found: {GENERATED_JSONL}"
    )


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    text = " ".join(text.split())

    return text.strip()


# =========================================================
# OLLAMA CALL
# =========================================================

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

            except json.JSONDecodeError:
                continue

    return full_response.strip()


# =========================================================
# FACTUAL ACCURACY SCORING
# =========================================================

def score_counter_speech(hate_speech, generated_cn):

    prompt = f"""
You are evaluating two counter speech responses
for the same hate speech.

Hate Speech:
{hate_speech}

Generated Counter Speech:
{generated_cn}

Evaluate BOTH responses independently
on a scale of 0 to 100.

Evaluation Criteria:
- factual accuracy
- relevance to the hate speech
- clarity and coherence
- usefulness in countering the hate speech
- respectful and constructive tone
- ability to counter harmful claims using factual information
- provides contextual clarification when discussing sensitive topics
- avoids reinforcing harmful stereotypes without correction

A response should receive LOWER scores if it:
- repeats harmful stereotypes without clarification
- reinforces discriminatory narratives
- presents harmful claims without factual correction
- promotes hostility, prejudice, or hate

A high-quality counter speech should:
- provide factual evidence, statistics, or historical clarification when appropriate
- constructively challenge misinformation or stereotypes
- encourage understanding and respectful dialogue

Return ONLY in this exact format:

Generated Score: <number>
"""

    result = call_ollama(prompt)

    score = None

    match = re.search(
        r"Generated Score:\s*(\d+(?:\.\d+)?)",
        result,
        re.IGNORECASE
    )

    if match:
        score = float(match.group(1))

        if not 0 <= score <= 100:
            score = None

    return score


# =========================================================
# LOAD GENERATED FILE
# =========================================================

print("\nLoading generated counter-speech file...\n")

generated_entries = []

with open(
    GENERATED_JSONL,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:
            generated_entries.append(
                json.loads(line)
            )

        except json.JSONDecodeError as e:
            print("JSON Error:", e)


# =========================================================
# SCORE ALL RESPONSES
# =========================================================

updated_entries = []

print("\nRunning factual accuracy scoring...\n")

for item in tqdm(generated_entries):

    hate_speech = normalize_text(
        item.get("hate_speech", "")
    )

    generated_cn = normalize_text(
        item.get("generated_counter_speech", "")
    )

    try:
        generated_score = score_counter_speech(
            hate_speech,
            generated_cn
        )

    except requests.RequestException as e:
        print(f"\nOllama request failed: {e}")
        generated_score = None

    updated_entries.append({

        "hate_speech": hate_speech,

        "generated_counter_speech": generated_cn,

        "generated_factual_accuracy": generated_score
    })


# =========================================================
# SAVE OUTPUT
# =========================================================

output_dir = os.path.dirname(OUTPUT_JSONL)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(
    OUTPUT_JSONL,
    "w",
    encoding="utf-8"
) as f:

    for item in updated_entries:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
                indent=2
            )
            + "\n\n"
        )


# =========================================================
# DONE
# =========================================================

print("\nFactual accuracy scoring completed.")

print("\nResults saved to:\n")

print(OUTPUT_JSONL)

print(f"\nTotal records processed: {len(updated_entries)}")
