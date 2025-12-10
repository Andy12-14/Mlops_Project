import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline


def derive_labels(scores: List[int]) -> List[int]:
    # Treat 4-5 stars as positive, else negative/neutral
    return [1 if s >= 4 else 0 for s in scores]


def evaluate(
    data_path: str,
    sample_size: int,
    threshold: float,
    output_path: str,
) -> float:
    df = pd.read_csv(data_path)
    df = df[df["content"].notna() & df["score"].notna()]

    if sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    texts = df["content"].tolist()
    truth = derive_labels(df["score"].astype(int).tolist())

    print(f"Running evaluation on {len(texts)} samples...")
    preds = clf(texts)
    pred_labels = [1 if p["label"] == "POSITIVE" else 0 for p in preds]

    acc = accuracy_score(truth, pred_labels)
    report = classification_report(truth, pred_labels, target_names=["negative", "positive"], output_dict=True)

    metrics = {
        "samples": len(texts),
        "accuracy": acc,
        "threshold": threshold,
        "classification_report": report,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output}")
    print(json.dumps(metrics, indent=2))

    if acc < threshold:
        raise SystemExit(
            f"Accuracy {acc:.3f} below threshold {threshold:.3f}. Failing evaluation."
        )

    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment model/pipeline on dataset")
    parser.add_argument("--data-path", default="dataset/dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--sample-size", type=int, default=200, help="Sample size for quick CI runs (0 for full)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum accuracy to pass")
    parser.add_argument("--output-path", default="metrics/metrics.json", help="Where to write metrics JSON")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    evaluate(
        data_path=args.data_path,
        sample_size=args.sample_size,
        threshold=args.threshold,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
