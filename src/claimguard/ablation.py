"""Ablation: how much does the LLM add? Everything here runs offline (no API calls).

Compares, on the same claims:
  1. v1 rule baseline        (rules.py, flags both copies of a duplicate)
  2. rules + duplicate ledger (the workflow's non-LLM part, scored with the same risk logic)
  3. full LLM pipeline        (only for the sample saved by pipeline_eval)

Run from the project root:
    python -m src.claimguard.ablation
"""
from pathlib import Path

import pandas as pd

from .doc_gen import CLAIMS_CSV
from .evaluate import CLAIMS_EASY_CSV, evaluate
from .extract import DATE_FIELDS, FIELDS
from .pipeline_eval import RESULTS_CSV
from .policy_agent import COSTS_CSV
from .workflow import Deps, make_nodes


def in_submission_order(claims: pd.DataFrame) -> pd.DataFrame:
    """Originals before copies, simulating arrival order (the data has no timestamps)."""
    late = (claims["fraud_type"] == "duplicate_claim").astype(int)
    return claims.assign(_late=late).sort_values(["_late", "claim_id"]).drop(columns="_late")


def run_rules_only(claims: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Rules + duplicate ledger + risk scoring, with no LLM at all."""
    nodes = make_nodes(Deps(costs, None, None))
    rows = []
    for rec in claims.to_dict("records"):
        extracted = {f: (str(rec[f]) if f in DATE_FIELDS else rec[f]) for f in FIELDS}
        state = {"claim_id": rec["claim_id"], "extracted": extracted}
        state.update(nodes["rules"](state))
        state.update(nodes["decide"](state))
        rows.append({"claim_id": rec["claim_id"], "is_fraud": bool(rec["is_fraud"]),
                     "fraud_type": rec["fraud_type"], "recommendation": state["recommendation"],
                     "risk_score": state["risk_score"], "signals": ",".join(state["signals"])})
    return pd.DataFrame(rows)


def metrics(res: pd.DataFrame) -> dict:
    flagged, fraud = res["recommendation"] != "approve", res["is_fraud"]
    tp, fp = int((fraud & flagged).sum()), int((~fraud & flagged).sum())
    fn, tn = int((fraud & ~flagged).sum()), int((~fraud & ~flagged).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def line(name: str, m: dict) -> str:
    return (f"  {name:34s} TP={m['TP']:3d} FP={m['FP']:3d} FN={m['FN']:3d} TN={m['TN']:3d}"
            f"  precision={m['precision']:.1%}  recall={m['recall']:.1%}")


def disagreements(llm_res: pd.DataFrame, rules_res: pd.DataFrame) -> pd.DataFrame:
    both = llm_res.merge(rules_res, on="claim_id", suffixes=("_llm", "_rules"))
    return both[(both["recommendation_llm"] != both["recommendation_rules"])
                | (both["signals_llm"].fillna("") != both["signals_rules"].fillna(""))]


if __name__ == "__main__":
    costs = pd.read_csv(COSTS_CSV)
    print("=== Whole dataset (500 claims, 15% fraud) ===")
    for name, path in [("EASY", CLAIMS_EASY_CSV), ("HARD", CLAIMS_CSV)]:
        claims = pd.read_csv(path)
        v1 = evaluate(claims)
        print(f"{name} data")
        print(line("1. v1 rules (no ledger)", {"TP": v1["tp"], "FP": v1["fp"], "FN": v1["fn"], "TN": v1["tn"],
                                               "precision": v1["precision"], "recall": v1["recall"]}))
        print(line("2. rules + ledger (no LLM)", metrics(run_rules_only(in_submission_order(claims), costs))))

    if Path(RESULTS_CSV).exists():
        llm = pd.read_csv(RESULTS_CSV)
        claims = pd.read_csv(CLAIMS_CSV).set_index("claim_id").loc[llm["claim_id"]].reset_index()
        rules_res = run_rules_only(claims, costs)       # same claims, same order as the LLM run
        print(f"\n=== Same {len(llm)} claims as the LLM pipeline run ===")
        llm["is_fraud"] = llm["is_fraud"].astype(bool)
        print(line("2. rules + ledger (no LLM)", metrics(rules_res)))
        print(line("3. full LLM pipeline", metrics(llm)))
        diff = disagreements(llm, rules_res)
        print(f"\nClaims where the two differ: {len(diff)}")
        for r in diff.to_dict("records"):
            print("  ", r["claim_id"], "| LLM:", r["recommendation_llm"], r["signals_llm"],
                  "| rules:", r["recommendation_rules"], r["signals_rules"])
    else:
        print("\n(run pipeline_eval first to compare against the LLM pipeline)")