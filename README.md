# Vita AI

Vita AI is a from-scratch Python prototype for transparent plant-health screening. It combines a
private photo-quality check with a symptom-led decision-support engine, shows the evidence behind
its shortlist, and explicitly abstains when the case is unclear.

It does **not** contain a trained plant-disease classifier and does **not** claim a field-accuracy
percentage. A rule-match percentage is an interpretable evidence score, not a probability or a
diagnosis.

## What the prototype includes

- Streamlit interface with Overview, New screening, History, Knowledge, and About pages
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
- No image storage and no account requirement

## Supported crops

Apple, Corn, Cucumber, Grape, Pepper, Potato, Strawberry, and Tomato.

## Run on Windows

Python 3.11 or newer is required.

```powershell
cd path\to\vita-ai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also run `setup_windows.bat` once, followed by `run_windows.bat`.

## Run quality checks

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The unit tests do not require Streamlit. They require only NumPy and Pillow.

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
│   └── data/conditions.json            Curated crop-condition rules
├── tests/                              Automated tests
├── requirements.txt
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
3. Train a model with explicit unknown/out-of-distribution handling.
4. Calibrate outputs on a locked field validation set.
5. Publish per-crop sensitivity, specificity, calibration error, abstention rate, and limitations.
6. Run prospective field evaluation before treatment guidance or high-impact decisions.

Until then, this prototype is appropriate for interface testing, workflow validation, knowledge-base
review, and gathering structured feedback—not autonomous agricultural diagnosis.

