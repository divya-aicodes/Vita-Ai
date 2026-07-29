"""Evaluate a trained model and generate dashboard-ready evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    load_class_names,
)
from src.model import load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Vita AI CNN.")
    parser.add_argument("--data", type=Path, required=True, help="Test folder containing one folder per class.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Test dataset directory does not exist: {args.data}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        top_k_accuracy_score,
    )

    labels = load_class_names()
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data,
        image_size=(224, 224),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
        class_names=labels,
    )
    if list(dataset.class_names) != labels:
        raise SystemExit(
            "Test folder class order does not match models/class_names.json. "
            "Use the exact same class folder names used for training."
        )
    model = load_trained_model(args.model)
    y_true = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])
    probabilities = np.asarray(model.predict(dataset, verbose=1))
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "samples": int(len(y_true)),
        "classes": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top_3_accuracy": float(
            top_k_accuracy_score(y_true, probabilities, k=min(3, len(labels)), labels=range(len(labels)))
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    frame = pd.DataFrame(report).transpose().reset_index(names="class")
    frame.to_csv(CLASSIFICATION_REPORT_PATH, index=False)

    matrix = confusion_matrix(y_true, y_pred)
    figure_size = max(9, min(22, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, cmap="Greens")
    ax.set(
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Confusion matrix · {len(y_true):,} test images",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=170)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote evaluation artifacts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
