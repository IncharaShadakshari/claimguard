\# ClaimGuard



Agentic AI system that triages health insurance claims and flags fraud risk, with human review for borderline cases.



\## Status

Work in progress. Currently building the synthetic dataset and a rule-based baseline.



\## Data

\- Claims are synthetic. No real patient data is used.

\- Procedure costs are approximate, based on published PM-JAY package rates where available and rough estimates otherwise.

\- Fraud patterns (duplicate claims, inflated bills, date mismatches, diagnosis-procedure mismatches, waiting-period violations) are injected synthetically.



\## Setup

&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt

