
import json
import statistics
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Input file path from environment configuration
INPUT_JSONL = os.getenv("INPUT_JSONL")

if not INPUT_JSONL:
    raise ValueError("INPUT_JSONL is not configured in the .env file")

generated_scores = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    content = f.read().strip()

    # Split on blank lines
    records = content.split("\n\n")

    for record in records:
        if not record.strip():
            continue

        try:
            data = json.loads(record)

            generated_scores.append(
                float(data["generated_factual_accuracy"])
            )

        except Exception as e:
            print("Error reading record:", e)

if not generated_scores:
    raise ValueError("No valid factual accuracy scores were found.")

generated_avg = statistics.mean(generated_scores)

print(f"Samples: {len(generated_scores)}")
print(f"Average Generated Factual Accuracy: {generated_avg:.2f}")
