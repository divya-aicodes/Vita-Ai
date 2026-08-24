# Changelog

All notable Vita AI changes are documented here.

## 1.4.0 — 2026-08-25

### Added

- Added required known-plant selection so screening candidates cannot belong to another crop.
- Added selected-plant support, within-plant disease scoring, separation checks, and an explicit `No reliable match` outcome.
- Added regression tests for cross-plant suppression, accepted matches, ambiguous matches, and unsupported crops.

### Changed

- Calibrated conservative acceptance thresholds on the 10,709-image controlled test split: 97.89% coverage and 99.98% accuracy among accepted results.
- Withholds disease guidance and Grad-CAM whenever a result fails reliability checks.
- Renamed image quality to photo quality and states that it measures technical image properties, not diagnostic accuracy.
- Updated reports and stored history to record withheld results as unknown instead of diagnoses.
- Bumped the application version to 1.4.0.

## 1.3.4 — 2026-08-14

### Changed

- Refined the cross-device sidebar opener with a lighter ivory/sage surface, forest arrow, and softer shadow.
- Preserved its 42×42 touch target and responsive behavior.
- Bumped the application version to 1.3.4.

## 1.3.3 — 2026-08-14

### Fixed

- Kept the sidebar opener available whenever navigation is collapsed on laptop, desktop, tablet, and mobile screens.
- Preserved the hidden fullscreen, Deploy, and Streamlit menu controls across all layouts.
- Bumped the application version to 1.3.3.

## 1.3.2 — 2026-08-14

### Fixed

- Restored a visible, touch-friendly mobile sidebar opener while keeping unrelated Streamlit toolbar actions hidden.
- Improved the sidebar close button contrast on the dark navigation surface.
- Bumped the application version to 1.3.2.

## 1.3.1 — 2026-08-14

### Changed

- Restored the original Vita AI leaf-and-lens logo inside the existing 46×46 sidebar brand mark without changing the sidebar layout.
- Bumped the application version to 1.3.1.

## 1.3.0 — 2026-08-14

### Added

- Added a Krea-generated botanical diagnostic hero illustration tailored to explainable plant-health AI.
- Added direct home-page actions, a capability trust strip, guided screening steps, improved empty states, and responsible-AI principle cards.

### Changed

- Redesigned the complete Streamlit interface with a premium forest, mineral sage, warm ivory, and chartreuse visual system.
- Improved navigation hierarchy, typography, spacing, controls, tables, metrics, expanders, mobile responsiveness, focus states, and reduced-motion behavior.
- Reworked page copy and information hierarchy to emphasize confidence, evidence, privacy, and responsible use.
- Replaced dependency on the deleted legacy logo assets with a resilient CSS wordmark and monogram.
- Bumped the application version to 1.3.0.

## 1.2.0 — 2026-08-04

### Added

- Added automated lint, compilation, and unit-test checks through GitHub Actions.
- Added Dependabot, structured issue forms, a pull-request checklist, contribution guidance, and a private security-reporting policy.
- Added SHA-256 and file-size verification for the downloadable model artifact.
- Added tests for model-artifact hashing and failed-database-transaction rollback.

### Changed

- Hardened SQLite with write-ahead logging, busy timeouts, explicit rollback, and foreign-key enforcement.
- Replaced unexpected internal exception details with safe user-facing messages and server-side logging.
- Standardized Python 3.12 setup, line endings, and Ruff quality rules.
- Bumped the application version to 1.2.0.

## 1.1.0 — 2026-08-04

### Added

- Expanded the Disease Library from 38 to 70 entries.
- Added 32 reference-only conditions across banana, rice, wheat, cotton, mango, onion, cucumber, chili pepper, peanut, papaya, coffee, and citrus.
- Added search across crops, conditions, categories, descriptions, and symptoms.
- Added crop, category, and model-coverage filters.
- Added library metrics, coverage badges, empty-state guidance, and authoritative reference links.

### Changed

- Clearly distinguishes CNN-supported prediction classes from educational reference entries.
- Bumped the application version to 1.1.0.
