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



\## LLM results (Groq, openai/gpt-oss-20b)



\*\*Document extraction\*\* (plain-text hospital documents, messy date and amount formats, "advance paid" distractor): 50/50 claims fully correct, including 40 fraud claims. The model copied impossible dates and wrong diagnoses faithfully instead of "fixing" them. Caveat: documents are clean typed text, not scans, so this is an easy test.



\*\*Policy agent\*\* (10-clause sample policy, BM25 retrieval, LLM cites violated clauses; 40 fraud + 10 clean claims):



| Fraud type | Flagged | Correct clause cited |

|---|---|---|

| Waiting period | 10/10 | 10/10 |

| Date mismatch | 10/10 | 10/10 |

| Diagnosis-procedure mismatch | 10/10 | 10/10 |

| Inflated billing | 8/10 | 8/10 |



0/10 clean claims flagged, 0 invented clause IDs (invented IDs are dropped and counted).



\*\*Honest notes:\*\* retrieval is trivial with only 10 clauses (always the same \~6 are retrieved). Claim facts such as day counts are computed in code, not by the LLM. Duplicates cannot be found by this agent and are handled by the rule check.



\## Full pipeline results (LangGraph workflow, Groq gpt-oss-20b)



Workflow: extract fields from the document, run rule checks (with a duplicate ledger), ask the policy agent which clauses are violated, combine signals into a risk score, and send anything above the approve threshold to a human reviewer. Nothing is ever auto-denied.



Evaluated on a stratified sample of 37 hard-mode claims (20 fraud, 17 clean):



| TP | FP | FN | TN | Precision | Recall |

|---|---|---|---|---|---|

| 20 | 2 | 0 | 15 | 90.9% | 100% |



All 20 fraud claims were flagged and the correct fraud signal was named each time. Date mismatches and duplicates scored high-risk (0.9); waiting-period, diagnosis mismatch and inflated billing went to human review (0.5-0.7).



\*\*Read this carefully:\*\*

\- The sample is enriched (54% fraud, plus 3 deliberately expensive legitimate claims). These numbers are not population estimates. At a realistic 15% fraud rate, precision would be lower.

\- Both false positives were legitimate claims billed 1.3-1.8x the usual charge. They scored 0.5, so they go to a human reviewer rather than being rejected.

\- The rule-only baseline missed 8 of 15 duplicates on hard data because of name typos. The workflow fixes this with a duplicate ledger that ignores the patient name. That improvement is a logic change, not an LLM gain. An ablation (rules + ledger, no LLM) is planned.

\- Signal weights are hand-set, not learned. Duplicates are evaluated in simulated submission order because the synthetic data has no timestamps.

\- Human review is demonstrated with LangGraph's interrupt/resume: the graph pauses on a flagged claim, shows the evidence, and records the reviewer's decision.



\## What went wrong (failure analysis)

\- \*\*Duplicates were flagged twice.\*\* Without a submission timestamp the rule flags both copies, so the genuine original becomes a false positive (7 on hard data).

\- \*\*Typos defeat exact matching.\*\* 8 of 15 duplicates were missed on hard data because the patient name differed by one letter.

\- \*\*A fixed threshold misfires on inflation.\*\* A 1.5x cost threshold wrongly flags 10 legitimate high-cost claims and misses 2 mild inflations.

\- Date, diagnosis-procedure and waiting-period fraud were caught 100% of the time, because those patterns are deterministic and the data is synthetic.



\## Roadmap

\- \[x] Synthetic data with ground-truth labels

\- \[x] Rule-based baseline and evaluation

\- \[x] LLM document extraction (bills, discharge summaries)

\- \[x] Policy clause retrieval (RAG)

\- \[x] Multi-agent workflow with human review

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

