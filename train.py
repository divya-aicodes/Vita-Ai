"""Train a baseline CNN or MobileNetV2 on a directory-structured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import (
    HISTORY_PATH,
    LABELS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    ensure_runtime_directories,
)
from src.model import build_custom_cnn, build_mobilenetv2, compile_model, runtime_metadata_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Vita AI plant-disease classifier.")
    parser.add_argument(
        "--data", type=Path, required=True, help="Dataset root or a folder containing train/valid."
    )
    parser.add_argument("--model", choices=("mobilenetv2", "custom-cnn"), default="mobilenetv2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--no-imagenet", action="store_true", help="Do not download ImageNet weights."
    )
    return parser.parse_args()


def resolve_data_folders(root: Path) -> tuple[Path, Path | None]:
    train_candidates = (root / "train", root / "Train", root / "training")
    valid_candidates = (root / "valid", root / "val", root / "validation", root / "Valid")
    train = next((path for path in train_candidates if path.is_dir()), None)
    valid = next((path for path in valid_candidates if path.is_dir()), None)
    return (train or root), valid


def load_datasets(args, tf):
    train_dir, valid_dir = resolve_data_folders(args.data)
    common = {
        "image_size": (224, 224),
        "batch_size": args.batch_size,
        "label_mode": "int",
        "seed": args.seed,
    }
    if valid_dir is None:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=args.validation_split,
            subset="training",
            shuffle=True,
            **common,
        )
        valid_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=args.validation_split,
            subset="validation",
            shuffle=False,
            **common,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, **common)
        valid_ds = tf.keras.utils.image_dataset_from_directory(valid_dir, shuffle=False, **common)
        if train_ds.class_names != valid_ds.class_names:
            raise ValueError("Training and validation class folders do not match.")
    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), valid_ds.prefetch(autotune), class_names


def callbacks(tf, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(output, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]


def merge_history(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def main() -> None:
    args = parse_args()
    if not args.data.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.data}")
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        raise SystemExit("Epoch counts must be non-negative and initial epochs must be at least 1.")
    ensure_runtime_directories()
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, valid_ds, class_names = load_datasets(args, tf)
    if len(class_names) < 2:
        raise SystemExit("At least two class folders are required.")

    if args.model == "mobilenetv2":
        model = build_mobilenetv2(
            len(class_names), trainable_backbone=False, imagenet_weights=not args.no_imagenet
        )
    else:
        model = build_custom_cnn(len(class_names))
    compile_model(model, learning_rate=1e-3)
    initial = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks(tf, args.output),
    )

    fine_tuned = None
    if args.model == "mobilenetv2" and args.fine_tune_epochs:
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        for layer in backbone.layers[:-55]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, learning_rate=1e-5)
        fine_tuned = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(initial.history["loss"]),
            callbacks=callbacks(tf, args.output),
        )

    model.save(args.output)
    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    metadata = runtime_metadata_for(model)
    METADATA_PATH.write_text(json.dumps(metadata.__dict__, indent=2), encoding="utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(merge_history(initial, fine_tuned), indent=2), encoding="utf-8"
    )
    print(f"Saved model to {args.output}")
    print(f"Saved {len(class_names)} class labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()
