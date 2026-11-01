"""Policy agent: retrieve relevant clauses, ask the LLM which ones a claim violates.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.claimguard.policy_agent --per-type 10 --clean 10
    python -m src.claimguard.policy_agent --retrieval-only      (free, no API calls)
"""
import argparse
import time

import pandas as pd

from .doc_gen import CLAIMS_CSV
from .extract import DEFAULT_MODEL, SLEEP_SECONDS, make_groq_llm, parse_json
from .policy_rag import claim_facts, retrieve_clauses

COSTS_CSV = CLAIMS_CSV.parents[1] / "procedure_costs.csv"

SYSTEM_PROMPT = """You assist a health insurance claims reviewer.
You receive facts about one claim and a numbered list of policy clauses.
Decide which clauses, if any, the facts violate or put in doubt.
Rules:
- Cite only clause IDs that appear in the list. Never invent a clause.
- Use only the facts given. Do not assume anything else.
- A clause with no matching problem in the facts must NOT be cited.
Return ONLY JSON: {"violations": [{"clause": "C2", "reason": "<one short sentence>"}]}
Return {"violations": []} if the claim has no problem."""

# Which clause a correct answer should cite, per fraud type.
# Duplicates need other claims to compare against, so this agent cannot catch them.
EXPECTED = {
    "waiting_period_violation": {"C2"},
    "date_mismatch": {"C1", "C7"},
    "diagnosis_procedure_mismatch": {"C4"},
    "inflated_billing": {"C5"},
}


def build_prompt(facts: list, clauses: list) -> str:
    fact_lines = "\n".join(f"- {f}" for f in facts)
    clause_lines = "\n".join(f"{cid}: {text}" for cid, text in clauses)
    return f"CLAIM FACTS:\n{fact_lines}\n\nPOLICY CLAUSES:\n{clause_lines}"


def run_agent(llm_fn, claim: dict, costs: pd.DataFrame):
    """Returns a dict with violations (valid clause IDs only), or None if the reply is unusable."""
    facts = claim_facts(claim, costs)
    clauses = retrieve_clauses(facts)
    allowed = {cid for cid, _ in clauses}
    try:
        data = parse_json(llm_fn(build_prompt(facts, clauses)))
        raw = data.get("violations", [])
        if not isinstance(raw, list):
            raise ValueError("violations is not a list")
    except (ValueError, TypeError, AttributeError):
        return None
    valid, invalid = [], 0
    for v in raw:
        cid = str(v.get("clause", "")).strip().upper() if isinstance(v, dict) else ""
        if cid in allowed:
            valid.append({"clause": cid, "reason": str(v.get("reason", ""))[:300]})
        else:
            invalid += 1          # hallucinated or irrelevant citation: dropped and counted
    return {"violations": valid, "invalid_citations": invalid,
            "retrieved": sorted(allowed), "facts": facts}


def retrieval_report(claims: pd.DataFrame, costs: pd.DataFrame) -> None:
    print("=== Retrieval check (offline, all claims) ===")
    for ftype, expected in EXPECTED.items():
        sub = claims[claims["fraud_type"] == ftype]
        hits = sum(bool({c for c, _ in retrieve_clauses(claim_facts(r, costs))} & expected)
                   for r in sub.to_dict("records"))
        print(f"  {ftype:30s} expected clause retrieved: {hits}/{len(sub)}")


def evaluate_agent(claims, costs, llm_fn, per_type, clean, sleep=0):
    parts = [claims[claims["fraud_type"] == t].head(per_type) for t in EXPECTED]
    parts.append(claims[~claims["is_fraud"]].head(clean))
    sample = pd.concat(parts)
    stats = {t: {"n": 0, "flagged": 0, "right_clause": 0} for t in EXPECTED}
    clean_n = clean_fp = fails = invalid = 0
    recs = sample.to_dict("records")
    for i, rec in enumerate(recs, 1):
        res = run_agent(llm_fn, rec, costs)
        if res is None:
            fails += 1
            print(f"  [{i}/{len(recs)}] {rec['claim_id']} PARSE FAIL")
        else:
            invalid += res["invalid_citations"]
            cited = {v["clause"] for v in res["violations"]}
            if rec["is_fraud"]:
                s = stats[rec["fraud_type"]]
                s["n"] += 1
                s["flagged"] += bool(cited)
                s["right_clause"] += bool(cited & EXPECTED[rec["fraud_type"]])
            else:
                clean_n += 1
                clean_fp += bool(cited)
            print(f"  [{i}/{len(recs)}] {rec['claim_id']} cited={sorted(cited) or '-'}")
        if sleep and i < len(recs):
            time.sleep(sleep)
    return {"stats": stats, "clean_n": clean_n, "clean_fp": clean_fp,
            "parse_failures": fails, "invalid_citations": invalid}


def print_eval(r: dict) -> None:
    print("\n=== Policy agent vs ground truth ===")
    for t, s in r["stats"].items():
        print(f"  {t:30s} n={s['n']:2d}  flagged={s['flagged']:2d}  cited correct clause={s['right_clause']:2d}")
    print(f"  clean claims wrongly flagged: {r['clean_fp']}/{r['clean_n']}")
    print(f"  parse failures: {r['parse_failures']}   invalid citations dropped: {r['invalid_citations']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()
    claims, costs = pd.read_csv(CLAIMS_CSV), pd.read_csv(COSTS_CSV)
    retrieval_report(claims, costs)
    if not args.retrieval_only:
        llm = make_groq_llm(args.model, SYSTEM_PROMPT)
        print_eval(evaluate_agent(claims, costs, llm, args.per_type, args.clean, SLEEP_SECONDS))