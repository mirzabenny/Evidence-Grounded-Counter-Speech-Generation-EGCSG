
import json
import os
import random
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Dataset paths from environment variables
DEV = os.getenv("DEV_DATASET_PATH")
TRAIN = os.getenv("TRAIN_DATASET_PATH")
TEST = os.getenv("TEST_DATASET_PATH")
OUTPUT = os.getenv("COMBINED_CORPUS_OUTPUT_PATH")

# Validate configuration
required_paths = {
    "DEV_DATASET_PATH": DEV,
    "TRAIN_DATASET_PATH": TRAIN,
    "TEST_DATASET_PATH": TEST,
    "COMBINED_CORPUS_OUTPUT_PATH": OUTPUT
}

for variable, path in required_paths.items():
    if not path:
        raise ValueError(
            f"Missing environment variable: {variable}. "
            "Please configure it in your .env file."
        )

# Check whether input dataset files exist
for path in [DEV, TRAIN, TEST]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")


# Loads documents from a JSONL file
def load_with_limit(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)

                items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "text": entry.get("text", "")
                })

            except json.JSONDecodeError:
                continue

    return items


# Combine documents from DEV, TRAIN and TEST datasets
corpus = []

corpus += load_with_limit(DEV)
corpus += load_with_limit(TRAIN)
corpus += load_with_limit(TEST)

# Shuffle the combined corpus
random.shuffle(corpus)

# Create output directory if it does not exist
output_dir = os.path.dirname(OUTPUT)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Save the combined corpus as JSONL
with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item) + "\n")

print(f"Combined corpus created successfully: {OUTPUT}")
print(f"Total documents: {len(corpus)}")
