"""Train and compare Vita AI's five approved image classifiers.

Expected dataset layout:

    data/plant-disease/
      train/<class name>/*.jpg
      val/<class name>/*.jpg
      test/<class name>/*.jpg

The test split is loaded only for final evaluation. Do not tune against it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vita_ai.ml.modeling import (  # noqa: E402
    build_classifier,
    compile_classifier,
    enable_fine_tuning,
)
from vita_ai.ml.registry import MODEL_SPECS, get_model_spec  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "ml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[item.key for item in MODEL_SPECS],
        choices=[item.key for item in MODEL_SPECS],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _keras() -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is missing. Install dependencies with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc
    return keras


def validate_dataset(data_dir: Path) -> tuple[str, ...]:
    split_classes: dict[str, tuple[str, ...]] = {}
    split_hashes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing required dataset directory: {split_dir}")
        classes = tuple(sorted(item.name for item in split_dir.iterdir() if item.is_dir()))
        if len(classes) < 2:
            raise SystemExit(f"{split_dir} must contain at least two class directories.")
        split_classes[split] = classes
        files = [path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            raise SystemExit(f"No supported images found in {split_dir}.")
        split_hashes[split] = {hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    if len(set(split_classes.values())) != 1:
        raise SystemExit("train, val, and test must contain the same class directories.")
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    if any(split_hashes[left] & split_hashes[right] for left, right in pairs):
        raise SystemExit("Identical image content appears in more than one dataset split.")
    return split_classes["train"]


def make_dataset(
    directory: Path,
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    keras = _keras()
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.prefetch(keras.backend.backend() == "tensorflow" and -1 or 1)


def class_weights(train_dir: Path, class_names: tuple[str, ...]) -> dict[int, float]:
    counts = []
    for name in class_names:
        count = sum(
            1 for path in (train_dir / name).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        counts.append(count)
    total = sum(counts)
    if not total or any(count == 0 for count in counts):
        raise SystemExit("Every training class must contain at least one supported image.")
    return {index: total / (len(counts) * count) for index, count in enumerate(counts)}


def callbacks(output_dir: Path, patience: int, checkpoint_name: str) -> list[Any]:
    keras = _keras()
    return [
        keras.callbacks.ModelCheckpoint(
            output_dir / checkpoint_name,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=max(1, patience // 2), factor=0.3, min_lr=1e-7
        ),
        keras.callbacks.CSVLogger(output_dir / "training.csv", append=True),
    ]


def evaluate_model(model: Any, dataset: Any, class_names: tuple[str, ...]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    inference_seconds = 0.0
    evaluated_images = 0
    warmed_up = False
    for images, labels in dataset:
        if not warmed_up:
            model.predict_on_batch(images)
            warmed_up = True
        started = time.perf_counter()
        probability_batches.append(np.asarray(model.predict_on_batch(images)))
        inference_seconds += time.perf_counter() - started
        evaluated_images += len(labels)
        true_batches.append(np.asarray(labels))
    y_true = np.concatenate(true_batches)
    probabilities = np.concatenate(probability_batches)
    y_pred = probabilities.argmax(axis=1)
    top_k = min(3, len(class_names))
    top_predictions = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top_k_accuracy = np.mean(
        [truth in predicted for truth, predicted in zip(y_true, top_predictions, strict=True)]
    )

    confidence = probabilities.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    calibration_error = 0.0
    boundaries = np.linspace(0, 1, 11)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            calibration_error += float(
                np.mean(in_bin) * abs(np.mean(correct[in_bin]) - np.mean(confidence[in_bin]))
            )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    specificities = []
    for index in range(len(class_names)):
        true_negative = (
            matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        )
        false_positive = matrix[:, index].sum() - matrix[index, index]
        specificities.append(true_negative / max(1, true_negative + false_positive))
    if len(class_names) == 2:
        macro_roc_auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        macro_roc_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=np.arange(len(class_names)),
            multi_class="ovr",
            average="macro",
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_roc_auc_ovr": float(macro_roc_auc),
        "matthews_correlation": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(class_names)))),
        "top_3_accuracy": float(top_k_accuracy),
        "expected_calibration_error": calibration_error,
        "inference_ms_per_image": inference_seconds * 1000 / evaluated_images,
        "test_images": int(len(y_true)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def write_evaluation(
    output_dir: Path, metrics: dict[str, Any], class_names: tuple[str, ...]
) -> None:
    report = metrics.pop("classification_report")
    matrix = metrics.pop("confusion_matrix")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "precision", "recall", "f1-score", "support"))
        for name in class_names:
            row = report[name]
            writer.writerow(
                (name, row["precision"], row["recall"], row["f1-score"], row["support"])
            )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("actual/predicted", *class_names))
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow((name, *row))


def train_one(
    args: argparse.Namespace, model_key: str, class_names: tuple[str, ...]
) -> dict[str, Any]:
    spec = get_model_spec(model_key)
    model_dir = args.output_dir / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    train = make_dataset(
        args.data_dir / "train",
        image_size=spec.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation = make_dataset(
        args.data_dir / "val",
        image_size=spec.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    weights = class_weights(args.data_dir / "train", class_names)
    model, backbone = build_classifier(spec, num_classes=len(class_names))
    compile_classifier(model, learning_rate=1e-3)
    frozen_checkpoint = "frozen.weights.h5"
    model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks(model_dir, args.patience, frozen_checkpoint),
    )
    model.load_weights(model_dir / frozen_checkpoint)
    frozen_val_loss = float(model.evaluate(validation, verbose=0)[0])

    if args.fine_tune_epochs:
        fine_tuned_checkpoint = "fine-tuned.weights.h5"
        enable_fine_tuning(backbone, trainable_fraction=0.25)
        compile_classifier(model, learning_rate=1e-5)
        model.fit(
            train,
            validation_data=validation,
            epochs=args.fine_tune_epochs,
            class_weight=weights,
            callbacks=callbacks(model_dir, args.patience, fine_tuned_checkpoint),
        )
        model.load_weights(model_dir / fine_tuned_checkpoint)
        fine_tuned_val_loss = float(model.evaluate(validation, verbose=0)[0])
        if frozen_val_loss < fine_tuned_val_loss:
            model.load_weights(model_dir / frozen_checkpoint)

    model_path = model_dir / "model.keras"
    model.save(model_path)
    test = make_dataset(
        args.data_dir / "test",
        image_size=spec.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    metrics = evaluate_model(model, test, class_names)
    metrics.update(
        {
            "key": spec.key,
            "model": spec.display_name,
            "parameters": int(model.count_params()),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        }
    )
    write_evaluation(model_dir, metrics.copy(), class_names)
    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size values must be positive.")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    keras = _keras()
    keras.utils.set_random_seed(args.seed)
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    class_names = validate_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one(args, key, class_names) for key in args.models]
    ranked = sorted(
        results,
        key=lambda item: (
            -item["macro_f1"],
            -item["balanced_accuracy"],
            item["inference_ms_per_image"],
        ),
    )
    leaderboard = {
        "selection_metric": "macro_f1",
        "tie_breakers": ["balanced_accuracy", "inference_ms_per_image"],
        "best_model": ranked[0]["key"],
        "class_names": class_names,
        "models": ranked,
    }
    (args.output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Best model by macro F1: {ranked[0]['model']}")
    print(f"Leaderboard: {args.output_dir / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
