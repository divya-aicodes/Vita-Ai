# ᕓ𐌉𐌕𐌀 𐌀𐌉

**An Explainable CNN-Based Plant Disease Screening and Advisory System**

Vita AI is a privacy-conscious Streamlit application for educational plant-leaf screening. It validates an image, checks its visual quality, passes a `224 × 224 × 3` RGB tensor to a CNN, shows confidence-aware top-three predictions, generates a Grad-CAM explanation, provides conservative disease information, and stores anonymous prediction metadata in SQLite.

It is a decision-support demonstration—not a confirmed agricultural diagnosis.

## Features

- JPG, JPEG, and PNG upload plus camera capture
- Genuine-image verification, EXIF orientation handling, and RGB conversion
- Blur, brightness, contrast, dimensions, and visible-content checks
- MobileNetV2 transfer-learning training pipeline
- Optional custom CNN academic baseline
- 38 PlantVillage crop/condition classes
- High, moderate, and low-confidence response rules
- Top-three predictions
- Grad-CAM explanation for models trained by this repository
- Disease overview, visible symptoms, and non-chemical prevention guidance
- Anonymous SQLite history, analytics, and optional user feedback
- Downloadable text screening report
- Evaluation dashboard generated from real artifacts only
- Responsive Streamlit interface
- React Bits-inspired ShinyText headings and pointer-responsive BorderGlow hero panels

## Quick start on Windows

### Already prepared installation

If the `.venv` folder and model are present, double-click:

```text
run_vita_ai.bat
```

### Fresh installation

Install Python 3.10 or 3.11 from <https://www.python.org/downloads/> and enable **Add Python to PATH** during installation. Then double-click:

```text
setup_windows.bat
```

The setup creates an isolated environment, installs the requirements, and downloads the optional compatible model. When it finishes, run `run_vita_ai.bat`.

### Command line

```powershell
cd "D:\Vita Ai"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL, usually <http://localhost:8501>.

## Accuracy and responsible use

No image classifier is “perfectly accurate.” Accuracy depends on:

- whether the crop and condition are among the 38 trained classes;
- focus, lighting, scale, background, and camera quality;
- the difference between controlled PlantVillage images and real field scenes;
- early or mixed symptoms;
- visually similar nutrient, insect, weather, and disease damage.

The optional downloadable model’s model card reports 98.75% test accuracy on an augmented PlantVillage test set. That is a third-party, self-reported dataset result—not a promise of field accuracy. The app deliberately does not add that number to its performance dashboard. Use `evaluate.py` with an untouched local test set to produce locally verified evidence.

This repository includes a local evaluation of that model on the official leaf-grouped PlantVillage colour test split: 10,709 images across 38 classes, with 99.18% accuracy, 99.95% top-three accuracy, and 98.83% macro F1. The generated metrics, class report, and confusion matrix are stored in `results/`. These controlled-dataset results still do not guarantee field performance.

## Dataset layout

Do not commit the dataset. A recommended layout is:

```text
data/
└── plantvillage/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   └── ...
    ├── valid/
    │   ├── Apple___Apple_scab/
    │   └── ...
    └── test/
        ├── Apple___Apple_scab/
        └── ...
```

You may also provide one folder containing class folders. `train.py` then creates a deterministic training/validation split.

## Train your own model

```powershell
.\.venv\Scripts\python.exe train.py `
  --data data\plantvillage `
  --model mobilenetv2 `
  --epochs 12 `
  --fine-tune-epochs 5
```

Training uses:

- realistic flip, small rotation, zoom, and contrast augmentation;
- ImageNet-pretrained MobileNetV2 by default;
- a frozen feature-extraction phase;
- optional fine-tuning of the top 55 backbone layers;
- early stopping, best-model checkpointing, and learning-rate reduction;
- sparse categorical cross-entropy, accuracy, and top-three accuracy.

For a no-network architecture test:

```powershell
.\.venv\Scripts\python.exe train.py --data data\small-test --no-imagenet --epochs 1 --fine-tune-epochs 0
```

## Evaluate the model

Keep the test folder untouched during training, then run:

```powershell
.\.venv\Scripts\python.exe evaluate.py --data data\plantvillage\test
```

This writes:

- `results/metrics.json`
- `results/training_history.json`
- `results/classification_report.csv`
- `results/confusion_matrix.png`

The app reads those artifacts on the **Model performance** page.

## Run automated tests

The core unit tests do not need TensorFlow or Streamlit:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover genuine-image validation, RGB conversion, CNN tensor shape, quality checks, label parsing, confidence boundaries, top-three ranking, SQLite persistence, feedback, analytics, and configuration consistency.

## Project structure

```text
Vita Ai/
├── app.py                       Streamlit application
├── train.py                     MobileNetV2/custom-CNN training
├── evaluate.py                  Real test-set evaluation artifacts
├── download_model.py            Optional compatible model installer
├── run_vita_ai.bat              One-click launcher
├── setup_windows.bat            Fresh Windows setup
├── requirements.txt
├── src/
│   ├── config.py
│   ├── database.py
│   ├── disease_info.py
│   ├── gradcam.py
│   ├── model.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── quality_checker.py
│   └── report.py
├── database/
│   └── schema.sql
├── models/
│   ├── class_names.json
│   └── model_metadata.json
├── results/
├── assets/
│   ├── vita-ai-logo.png          Primary square logo
│   ├── vita-ai-icon.png          Browser/app icon
│   ├── vita-ai-logo-transparent.png
│   └── sample_images/
└── tests/
```

## Privacy and security

- No login or personal information is requested.
- Uploaded files are decoded as genuine JPEG/PNG images and capped at 10 MB.
- Uploaded images are processed in memory and are not stored.
- SQLite contains prediction metadata and optional feedback only.
- Users can clear local history from the app.
- The app does not expose local file paths to uploaders.
- It gives no pesticide product or dosage recommendations.

## Troubleshooting

### “TensorFlow is not installed”

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### “No trained model was found”

Run:

```powershell
.\.venv\Scripts\python.exe download_model.py
```

or train a model with `train.py`.

### Model and labels do not match

The order and count in `models/class_names.json` must exactly match the model output layer. Never reorder labels after training.

### Grad-CAM unavailable

Grad-CAM is guaranteed for models created by this repository’s model builders. A third-party Keras model may use different internal layer names; prediction will still work, but the app will report explainability as unavailable instead of crashing.

## References

- TensorFlow MobileNetV2: <https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2>
- TensorFlow transfer learning guide: <https://www.tensorflow.org/tutorials/images/transfer_learning>
- Keras Grad-CAM example: <https://keras.io/examples/vision/grad_cam/>
- Optional model: <https://huggingface.co/rarfileexe/Plant-Disease-Detector>

## Animation integration

The supplied React Bits `ShinyText` and `BorderGlow` behavior is integrated through
`src/animations.py`. Because Streamlit is Python-based and does not expose its
internal React tree, the component behavior is translated into isolated HTML,
CSS, and JavaScript rather than adding an unnecessary second React build.
The effects include:

- a repeating directional shine across page headings;
- pointer-sensitive edge proximity and cursor angle;
- a multi-layer mesh-gradient border and outer glow;
- an introductory border sweep;
- responsive sizing and `prefers-reduced-motion` accessibility.
