import json
import statistics
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Input JSONL file containing evaluation metrics
INPUT_JSONL = os.getenv("AVERAGE_METRICS_INPUT_JSONL")

if not INPUT_JSONL:
    raise ValueError(
        "AVERAGE_METRICS_INPUT_JSONL is not configured in the .env file"
    )



rouge_precision_scores = []
rouge_recall_scores = []
rouge_fmeasure_scores = []
bleu_scores = []
bert_scores = []
factual_scores = []
toxicity_scores = []




with open(INPUT_JSONL, "r", encoding="utf-8") as f:

    content = f.read().strip()

    entries = content.split("\n\n")

    for entry in entries:

        if not entry.strip():
            continue

        try:

            data = json.loads(entry)

            scores = data.get("scores", {})

         

            rouge_text = scores.get(
                "rouge2",
                ""
            )

            precision_match = re.search(
                r"precision=([0-9.]+)",
                rouge_text
            )

            recall_match = re.search(
                r"recall=([0-9.]+)",
                rouge_text
            )

            fmeasure_match = re.search(
                r"fmeasure=([0-9.]+)",
                rouge_text
            )

            if precision_match:

                rouge_precision_scores.append(
                    float(
                        precision_match.group(1)
                    )
                )

            if recall_match:

                rouge_recall_scores.append(
                    float(
                        recall_match.group(1)
                    )
                )

            if fmeasure_match:

                rouge_fmeasure_scores.append(
                    float(
                        fmeasure_match.group(1)
                    )
                )



            bleu = scores.get("bleu2")

            if bleu is not None:

                bleu_scores.append(
                    float(bleu)
                )




            bert = scores.get("bert_f1")

            if bert is not None:

                bert_scores.append(
                    float(bert)
                )



           

            factual = scores.get(
                "factual_accuracy"
            )

            if factual is not None:

                factual_scores.append(
                    float(factual)
                )



      

            toxicity = scores.get(
                "toxicity"
            )

            if toxicity is not None:

                toxicity_scores.append(
                    float(toxicity)
                )

        except Exception as e:

            print(
                "Error reading entry:",
                e
            )
            
if rouge_precision_scores:

    print(
        "Average ROUGE2 Precision:",
        round(
            statistics.mean(
                rouge_precision_scores
            ),
            4
        )
    )

if rouge_recall_scores:

    print(
        "Average ROUGE2 Recall:",
        round(
            statistics.mean(
                rouge_recall_scores
            ),
            4
        )
    )

if rouge_fmeasure_scores:

    print(
        "Average ROUGE2 F1:",
        round(
            statistics.mean(
                rouge_fmeasure_scores
            ),
            4
        )
    )



if bleu_scores:

    print(
        "Average BLEU2:",
        round(
            statistics.mean(
                bleu_scores
            ),
            4
        )
    )




if bert_scores:

    print(
        "Average BERT F1:",
        round(
            statistics.mean(
                bert_scores
            ),
            4
        )
    )


if factual_scores:

    print(
        "Average Factual Accuracy:",
        round(
            statistics.mean(
                factual_scores
            ),
            4
        )
    )




if toxicity_scores:

    print(
        "Average Toxicity:",
        round(
            statistics.mean(
                toxicity_scores
            ),
            4
        )
    )

