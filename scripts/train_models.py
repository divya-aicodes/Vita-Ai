"""Train, evaluate, and export Vita AI's five PlantVillage classifiers."""

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
    data_config,
    enable_fine_tuning,
    freeze_backbone,
    trainable_parameters,
)
from vita_ai.ml.registry import MODEL_SPECS, get_model_spec  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def libraries() -> tuple[Any, Any, Any]:
    try:
        import timm
        import torch
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            "The ML environment is missing. Install requirements-ml.txt and a CUDA-enabled "
            "PyTorch build before training."
        ) from exc
    return torch, torchvision, timm


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
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_dataset(data_dir: Path) -> tuple[str, ...]:
    split_classes: dict[str, tuple[str, ...]] = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing required dataset directory: {split_dir}")
        classes = tuple(sorted(item.name for item in split_dir.iterdir() if item.is_dir()))
        if len(classes) != 38:
            raise SystemExit(f"{split_dir} must contain the 38 PlantVillage class directories.")
        split_classes[split] = classes
        files = [path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            raise SystemExit(f"No supported images found in {split_dir}.")
    if len(set(split_classes.values())) != 1:
        raise SystemExit("train, val, and test must contain identical class directories.")
    metadata_path = data_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit("dataset_metadata.json is required to prove the prepared split policy.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("class_count") != 38 or metadata.get("counts", {}).get("test") != 10709:
        raise SystemExit("Dataset metadata does not match the expected PlantVillage split.")
    return split_classes["train"]


def select_device(requested: str) -> Any:
    torch, _, _ = libraries()
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def create_loaders(
    data_dir: Path,
    model: Any,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[Any, Any, Any, tuple[str, ...], dict[str, Any]]:
    torch, torchvision, timm = libraries()
    config = data_config(model)
    train_transform = timm.data.create_transform(**config, is_training=True)
    evaluation_transform = timm.data.create_transform(**config, is_training=False)
    datasets = {
        "train": torchvision.datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": torchvision.datasets.ImageFolder(data_dir / "val", transform=evaluation_transform),
        "test": torchvision.datasets.ImageFolder(data_dir / "test", transform=evaluation_transform),
    }
    if len({tuple(dataset.classes) for dataset in datasets.values()}) != 1:
        raise SystemExit("Dataset loaders resolved inconsistent class orders.")
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        name: torch.utils.data.DataLoader(
            dataset,
            shuffle=name == "train",
            drop_last=name == "train",
            **common,
        )
        for name, dataset in datasets.items()
    }
    return (
        loaders["train"],
        loaders["val"],
        loaders["test"],
        tuple(datasets["train"].classes),
        config,
    )


def class_weights(loader: Any, device: Any) -> Any:
    torch, _, _ = libraries()
    counts = np.bincount(loader.dataset.targets, minlength=len(loader.dataset.classes))
    if np.any(counts == 0):
        raise SystemExit("Every PlantVillage class must have training examples.")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    *,
    optimizer: Any | None,
    scaler: Any | None,
) -> tuple[float, float]:
    torch, _, _ = libraries()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ),
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_examples += len(labels)
        if training and batch_index % 200 == 0:
            print(
                f"  batch {batch_index:,}/{len(loader):,} "
                f"loss={total_loss / total_examples:.4f} "
                f"accuracy={total_correct / total_examples:.2%}",
                flush=True,
            )
    return total_loss / total_examples, total_correct / total_examples


def train_phase(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    criterion: Any,
    device: Any,
    output_dir: Path,
    *,
    phase: str,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[Path, float]:
    torch, _, _ = libraries()
    checkpoint = output_dir / f"{phase}.pt"
    optimizer = torch.optim.AdamW(trainable_parameters(model), learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=1, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
        )
        scheduler.step(validation_loss)
        print(
            f"{phase} epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.2%} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.2%} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                print(f"Early stopping {phase} after epoch {epoch}", flush=True)
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return checkpoint, best_loss


def evaluate_model(
    model: Any, loader: Any, device: Any, class_names: tuple[str, ...]
) -> dict[str, Any]:
    torch, _, _ = libraries()
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

    model.eval()
    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    inference_seconds = 0.0
    evaluated_images = 0
    warmed_up = False
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if not warmed_up:
                model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warmed_up = True
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            evaluated_images += len(labels)
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            true_batches.append(labels.numpy())

    y_true = np.concatenate(true_batches)
    probabilities = np.concatenate(probability_batches)
    y_pred = probabilities.argmax(axis=1)
    top_predictions = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    top_3_accuracy = np.mean(
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

    labels = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for index in labels:
        true_negative = (
            matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        )
        false_positive = matrix[:, index].sum() - matrix[index, index]
        specificities.append(true_negative / max(1, true_negative + false_positive))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_roc_auc_ovr": float(
            roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
        ),
        "matthews_correlation": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "top_3_accuracy": float(top_3_accuracy),
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


def export_onnx(model: Any, model_path: Path, config: dict[str, Any]) -> None:
    torch, _, _ = libraries()
    model = model.cpu().eval()
    input_size = tuple(config["input_size"])
    example = torch.zeros((1, *input_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        model_path,
        input_names=("image",),
        output_names=("logits",),
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(model_path))


def train_one(
    args: argparse.Namespace,
    model_key: str,
    expected_classes: tuple[str, ...],
    device: Any,
) -> dict[str, Any]:
    torch, _, _ = libraries()
    spec = get_model_spec(model_key)
    model_dir = args.output_dir / spec.key
    metrics_path = model_dir / "metrics.json"
    model_path = model_dir / "model.onnx"
    if args.resume and metrics_path.exists() and model_path.exists():
        print(f"Skipping completed model: {spec.display_name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {spec.display_name} ===", flush=True)
    model = build_classifier(spec, num_classes=len(expected_classes)).to(device)
    train_loader, validation_loader, test_loader, class_names, config = create_loaders(
        args.data_dir,
        model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if class_names != expected_classes:
        raise SystemExit(f"Unexpected class order for {spec.display_name}.")
    (model_dir / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    weights = class_weights(train_loader, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    freeze_backbone(model)
    frozen_checkpoint, frozen_loss = train_phase(
        model,
        train_loader,
        validation_loader,
        criterion,
        device,
        model_dir,
        phase="frozen",
        epochs=args.epochs,
        learning_rate=1e-3,
        patience=args.patience,
    )

    selected_checkpoint = frozen_checkpoint
    if args.fine_tune_epochs:
        enable_fine_tuning(model, trainable_fraction=0.25)
        fine_checkpoint, fine_loss = train_phase(
            model,
            train_loader,
            validation_loader,
            criterion,
            device,
            model_dir,
            phase="fine-tuned",
            epochs=args.fine_tune_epochs,
            learning_rate=2e-5,
            patience=args.patience,
        )
        if frozen_loss < fine_loss:
            selected_checkpoint = frozen_checkpoint
            model.load_state_dict(
                torch.load(frozen_checkpoint, map_location=device, weights_only=True)
            )
        else:
            selected_checkpoint = fine_checkpoint

    metrics = evaluate_model(model, test_loader, device, class_names)
    export_onnx(model, model_path, config)
    metadata = {
        "key": spec.key,
        "model": spec.display_name,
        "timm_name": spec.timm_name,
        "input_size": list(config["input_size"]),
        "mean": list(config["mean"]),
        "std": list(config["std"]),
        "crop_pct": float(config.get("crop_pct", 1.0)),
        "interpolation": config["interpolation"],
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "selected_checkpoint": selected_checkpoint.name,
    }
    metrics.update(metadata)
    write_evaluation(model_dir, metrics.copy(), class_names)
    print(
        f"{spec.display_name}: test_accuracy={metrics['accuracy']:.2%} "
        f"macro_f1={metrics['macro_f1']:.2%}",
        flush=True,
    )
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"classification_report", "confusion_matrix"}
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size values must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch, _, _ = libraries()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = select_device(args.device)
    print(f"Training device: {device}", flush=True)
    class_names = validate_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [train_one(args, key, class_names, device) for key in args.models]
    ranked = sorted(
        results,
        key=lambda item: (
            -item["macro_f1"],
            -item["balanced_accuracy"],
            item["inference_ms_per_image"],
        ),
    )
    dataset_metadata_path = args.data_dir / "dataset_metadata.json"
    dataset_metadata = (
        json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        if dataset_metadata_path.exists()
        else None
    )
    leaderboard = {
        "dataset": dataset_metadata,
        "selection_metric": "macro_f1",
        "tie_breakers": ["balanced_accuracy", "inference_ms_per_image"],
        "best_model": ranked[0]["key"],
        "class_names": class_names,
        "models": ranked,
    }
    (args.output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Best model by macro F1: {ranked[0]['model']}", flush=True)


if __name__ == "__main__":
    main()
