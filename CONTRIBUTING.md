# Contributing to Vita AI

Thank you for helping improve Vita AI. Keep changes focused, evidence-based, and safe for an educational plant-health screening tool.

## Development setup

Use Python 3.12 and an isolated virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Before opening a pull request, run:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Pull-request standards

- Explain the user need and the visible behavior change.
- Add or update tests for behavior changes.
- Keep the 38 model output labels in their trained order.
- Mark educational library entries as reference-only unless the CNN was retrained and evaluated for them.
- Do not add pesticide brands, application dosages, or claims of confirmed diagnosis.
- Do not commit datasets, uploaded images, trained weights, databases, secrets, or personal information.
- Include screenshots for interface changes and update the changelog for user-visible releases.

By contributing, you agree that your contribution may be distributed under the repository's applicable license terms.
