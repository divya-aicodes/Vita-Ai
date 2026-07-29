# Evaluation artifacts

This folder contains locally generated evaluation evidence for the optional PlantVillage model. The included run used the official leaf-grouped colour test split: 10,709 images across 38 classes.

- Accuracy: 99.18%
- Top-three accuracy: 99.95%
- Macro precision: 99.10%
- Macro recall: 98.65%
- Macro F1: 98.83%

The measurements describe performance on controlled PlantVillage images and do not guarantee results on field photographs. Run `python evaluate.py --data <test-folder>` to reproduce or replace the artifacts:

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.png`

`train.py` also creates `training_history.json`.
