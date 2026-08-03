# Security Policy

## Supported version

Security fixes are applied to the current `main` branch.

## Reporting a vulnerability

Report security issues privately through GitHub Security Advisories for this repository. Do not open a public issue containing credentials, exploit details or sensitive runtime data.

Include the following information when possible:

- affected file or workflow;
- impact and realistic attack scenario;
- reproduction steps;
- suggested mitigation;
- whether secrets or generated artifacts may have been exposed.

## Security-sensitive areas

Pay particular attention to:

- GitHub Actions token permissions;
- third-party market and news responses;
- dependency updates;
- HTML generation and dashboard output;
- model and state artifact deserialization;
- accidental credential or database commits.

Python serialization formats such as `joblib` and `pickle` can execute code while loading. Only load model and adaptive-state artifacts produced by trusted project workflows.

## Trading scope

This repository is designed for research and paper trading. It does not place live orders and should not be modified to handle exchange credentials without a separate security review.
