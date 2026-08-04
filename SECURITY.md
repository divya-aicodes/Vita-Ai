# Security policy

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use the repository's [private vulnerability reporting form](https://github.com/divya-aicodes/Vita-Ai/security/advisories/new) and include:

- the affected version or commit;
- the smallest reliable reproduction steps;
- the expected and observed impact;
- any suggested mitigation, if known.

Do not include real user images, credentials, tokens, or other personal information. A maintainer should acknowledge a complete report within seven days and coordinate disclosure after a fix is available.

## Supported versions

Security fixes target the current `main` branch and the currently deployed Streamlit Community Cloud version.

## Scope

Reports about upload validation, dependency integrity, model artifact delivery, accidental data exposure, or unauthorized database access are in scope. Model misclassification on its own is a documented limitation rather than a security vulnerability, but reproducible attacks that bypass validation or expose data are in scope.
