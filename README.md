\# ClaimGuard



Agentic AI system that triages health insurance claims and flags fraud risk. It never auto-denies a claim: suspicious claims go to a human reviewer with the evidence attached.



\*\*Status:\*\* work in progress. Done: synthetic data, rule-based baseline, evaluation. Next: LLM document extraction, policy RAG, agent workflow, demo.



\## Data

\- All claims are \*\*synthetic\*\*. No real patient data is used.

\- Procedure costs are approximate, based on published PM-JAY package rates where available and rough estimates otherwise.

\- 500 claims per dataset, 15% fraud, five injected fraud types: duplicate claim, inflated billing, date mismatch, diagnosis-procedure mismatch, waiting-period violation.

\- Two versions are generated:

&#x20; - \*\*Easy:\*\* obvious fraud (bills 2-4x typical cost, exact duplicates).

&#x20; - \*\*Hard:\*\* subtler fraud (bills 1.3-2.5x), duplicates with name typos, and 5% of legitimate claims with high bills (complications).

\- Assumptions: 30-day waiting period, no waiting period for fractures.



\## Baseline results (rules only, no ML or LLM)



| Dataset | Precision | Recall | F1 | False positive rate |

|---|---|---|---|---|

| Easy | 83.3% | 100.0% | 90.9% | 3.5% |

| Hard | 79.3% | 86.7% | 82.8% | 4.0% |



\*\*Read this carefully:\*\* the rules and the fraud generator were written by the same person, so the easy-data score is circular. It shows the pipeline works, not real-world accuracy. The hard data is the meaningful test.



\## What went wrong (failure analysis)

\- \*\*Duplicates were flagged twice.\*\* Without a submission timestamp the rule flags both copies, so the genuine original becomes a false positive (7 on hard data).

\- \*\*Typos defeat exact matching.\*\* 8 of 15 duplicates were missed on hard data because the patient name differed by one letter.

\- \*\*A fixed threshold misfires on inflation.\*\* A 1.5x cost threshold wrongly flags 10 legitimate high-cost claims and misses 2 mild inflations.

\- Date, diagnosis-procedure and waiting-period fraud were caught 100% of the time, because those patterns are deterministic and the data is synthetic.



\## Roadmap

\- \[x] Synthetic data with ground-truth labels

\- \[x] Rule-based baseline and evaluation

\- \[ ] LLM document extraction (bills, discharge summaries)

\- \[ ] Policy clause retrieval (RAG)

\- \[ ] Multi-agent workflow with human review

\- \[ ] Streamlit demo and deployment



\## Run it

&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt

&#x20;   python -m src.claimguard.data\_gen

&#x20;   python -m src.claimguard.evaluate

&#x20;   python -m pytest -q



\## Limitations

This tool assists human reviewers and does not replace them. Results come from synthetic data and must not be read as real-world fraud detection accuracy.

