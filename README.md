# Vita AI

[![Vita AI CI](https://github.com/divya-aicodes/Vita-Ai/actions/workflows/ci.yml/badge.svg)](https://github.com/divya-aicodes/Vita-Ai/actions)
[![PlantVillage models](https://img.shields.io/badge/models-PlantVillage%20v1-286444)](https://github.com/divya-aicodes/Vita-Ai/releases/tag/plantvillage-v1)

Vita AI is a from-scratch Python prototype for transparent plant-health screening. It combines a
private photo-quality check with a symptom-led decision-support engine, shows the evidence behind
its shortlist, and explicitly abstains when the case is unclear.

It also includes five working PlantVillage transfer-learning classifiers. The Streamlit app can
download any selected, versioned ONNX model from GitHub Releases, verify its checksum, cache it,
and return the top three of 38 classes. A rule-match percentage remains a separate interpretable
evidence score, not a model probability or a diagnosis.

## What the prototype includes

- Streamlit interface with Overview, New screening, History, Knowledge, Model lab, and About pages
- Safe JPEG, PNG, and WebP validation with an 8 MB upload limit
- Local photo checks for resolution, brightness, contrast, focus, and visible leaf coverage
- Transparent crop-specific symptom matching with conservative abstention
- 25 curated condition patterns across 8 crops
- Top-three shortlist with matched and missing evidence
- Urgency-aware next steps and prevention guidance without pesticide products or doses
- Anonymous SQLite history and optional feedback
- Downloadable text report
- Unit tests for screening, photo handling, the knowledge base, and persistence
- GitHub Actions quality, test, and compilation pipeline
- Comparable EfficientNetV2B0, ConvNeXtTiny, DenseNet121, MobileNetV3Large, and ResNet50V2 trainers
- User-selectable ONNX inference for all five trained models in the Streamlit screening workflow
- Test-set accuracy, balanced accuracy, macro/weighted F1, precision, recall, MCC, log loss,
  confusion matrix, model size, and inference-time reporting
- No image storage and no account requirement

## Supported crops

The transparent symptom-matching workflow covers 8 crops: Apple, Corn, Cucumber, Grape, Pepper,
Potato, Strawberry, and Tomato.

The ML classifiers cover all 14 crop categories in the 38-class PlantVillage color dataset: Apple,
Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash,
Strawberry, and Tomato. The two scopes are shown separately in the interface.

## Run on Windows

Python 3.11 or newer is required.

```powershell
cd path\to\vita-ai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also run `setup_windows.bat` once, followed by `run_windows.bat`.

## Use the five classifiers

1. Open **New screening** and upload a JPEG, PNG, or WebP leaf image.
2. Choose any of the five models under **ML disease classification**.
3. Select **Run selected ML model** to see its top-three PlantVillage predictions.
4. Open **Model lab** to compare the locked-test metrics and engineering trade-offs.

The first prediction with a model downloads its ONNX archive from the
[PlantVillage v1 GitHub Release](https://github.com/divya-aicodes/Vita-Ai/releases/tag/plantvillage-v1).
Vita AI checks the configured file size and SHA-256 digest before safe extraction, then caches the
model under `runtime/models/`. Later predictions reuse the cached model.

All five models independently perform the same 38-class task. They are alternatives for comparison,
not five sequential stages, and the user can run any one of them.

## Run quality checks

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The unit tests do not require Streamlit. They require only NumPy and Pillow.

## Train and compare the five models

Install the optional ML dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Download the PlantVillage source files (`data.zip`, `splits/`, and `leaf_grouping/`) from the
[Hugging Face dataset](https://huggingface.co/datasets/mohanty/PlantVillage) into
`data/plantvillage-source`, then prepare the authors' color split
(54,305 images, 38 classes):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_plantvillage.py `
  --source-dir data\plantvillage-source `
  --output-dir data\plantvillage
```

The script keeps the published 10,709-image test set locked and selects validation leaf groups
only from the published training split. The resulting split is 36,958 train, 6,638 validation,
and 10,709 test images. Every split contains the same class directories:

```text
data/plantvillage/
├── train/<class name>/*.{jpg,jpeg,png,webp}
├── val/<class name>/*.{jpg,jpeg,png,webp}
└── test/<class name>/*.{jpg,jpeg,png,webp}
```

Related photographs from the same plant, farm, or capture session must remain in one split. Do not
select hyperparameters using `test`; it is reserved for the final comparison.

Train all five approved models:

```powershell
.\.venv\Scripts\python.exe scripts\train_models.py `
  --data-dir data\plantvillage `
  --epochs 4 `
  --fine-tune-epochs 6
```

Generated weights, labels, per-class reports, confusion matrices, and `leaderboard.json` are written
under `artifacts/ml/`, which is intentionally ignored by Git. The winner is selected first by test
macro F1, then balanced accuracy, then inference time. `scripts/package_models.py` creates the
release archives, checksums, deployment manifest, and committed Model lab leaderboard.

Each model has a different engineering role:

- **ResNet50V2:** highest unrounded macro F1 and balanced accuracy; largest and slowest candidate
- **ConvNeXtTiny:** essentially tied on F1 with ResNet while using a substantially smaller export
- **EfficientNetV2B0:** strong size, speed, and accuracy balance
- **MobileNetV3Large:** smallest and fastest candidate for constrained deployment
- **DenseNet121:** feature-reuse baseline retained for transparent comparison

Locked PlantVillage test results (10,709 images):

| Model | Accuracy | Macro F1 | Balanced accuracy | GPU latency | ONNX size |
|---|---:|---:|---:|---:|---:|
| ResNet50V2 | 99.40% | 99.28% | 99.36% | 8.53 ms/image | 179.9 MB |
| ConvNeXtTiny | 99.38% | 99.28% | 99.25% | 2.59 ms/image | 106.3 MB |
| EfficientNetV2B0 | 96.67% | 95.58% | 96.70% | 0.74 ms/image | 22.6 MB |
| MobileNetV3Large | 94.39% | 92.78% | 94.16% | 0.60 ms/image | 16.2 MB |
| DenseNet121 | 87.52% | 83.72% | 85.97% | 2.12 ms/image | 27.1 MB |

ResNet50V2 ranks first by unrounded macro F1; MobileNetV3Large is the smallest and fastest.
Latency was measured on the training machine's RTX 4060 Laptop GPU and is not a Streamlit CPU
latency guarantee.

The dataset determines the supported classes. A PlantVillage result is a controlled-dataset result;
it must not be described as field-photo accuracy without a separate representative field test set.
Complete precision, recall, specificity, ROC-AUC, MCC, log-loss, top-three accuracy, calibration,
per-class reports, and confusion matrices are generated by `scripts/train_models.py`. The deployed
summary is stored in `src/vita_ai/data/model_leaderboard.json`.

## Streamlit deployment

The deployed app should track the repository's `main` branch and run `app.py`. Large model binaries
are deliberately excluded from Git; the committed `model_manifest.json` points to immutable release
assets instead. Streamlit needs only `requirements.txt`, which includes ONNX Runtime for CPU
inference. No API key or private model-storage credential is required.

## Architecture

```text
vita-ai/
├── app.py                              Streamlit user interface
├── src/vita_ai/
│   ├── config.py                       Limits and screening vocabulary
│   ├── domain.py                       Typed result objects
│   ├── image_quality.py                Secure decoding and photo assessment
│   ├── knowledge.py                    Knowledge-base validation
│   ├── screening.py                    Transparent ranking and abstention
│   ├── storage.py                      SQLite case metadata and feedback
│   ├── report.py                       Downloadable report generation
│   ├── ml/                              Training, secure model delivery, and ONNX inference
│   └── data/conditions.json            Curated crop-condition rules
├── scripts/prepare_plantvillage.py       Reproducible grouped dataset preparation
├── scripts/train_models.py               Five-model training and locked-test evaluation
├── scripts/package_models.py             Verified GitHub Release packaging
├── tests/                              Automated tests
├── requirements.txt
├── requirements-ml.txt
└── requirements-dev.txt
```

The image is decoded and assessed in memory. It is never inserted into SQLite. Only the selected
crop, evidence, outcome, quality status, notes, and optional feedback are stored locally.

## How matching works

For the selected crop, each condition receives a bounded evidence score composed of:

- 72% visible symptom agreement
- 13% environment agreement
- 10% location/distribution agreement
- 5% broad photo observations

The photo contribution is deliberately weak because colour ratios are not disease-specific. A
single condition is presented as the closest screening match only when it clears the minimum score,
evidence-count, and lead-over-runner-up gates. Otherwise Vita AI returns `No reliable single match`.

These thresholds are product-safety defaults, not statistically calibrated thresholds.

## Production roadmap

A production disease classifier should be added only after the following evidence exists:

1. Collect consented, expert-labelled field images for each intended crop and region.
2. Split data by farm, plant, and capture session so related images cannot leak between splits.
3. Compare the five candidate models and add explicit unknown/out-of-distribution handling.
4. Calibrate outputs on a locked field validation set.
5. Publish per-crop sensitivity, specificity, calibration error, abstention rate, and limitations.
6. Run prospective field evaluation before treatment guidance or high-impact decisions.

Until then, this prototype is appropriate for interface testing, workflow validation, knowledge-base
review, and gathering structured feedback—not autonomous agricultural diagnosis.

