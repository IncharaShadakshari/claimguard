"""Run the full workflow on a stratified sample and measure it against the ground truth.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.claimguard.pipeline_eval --fraud-per-type 4 --clean 10
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from .data_gen import FRAUD_TYPES
from .doc_gen import CLAIMS_CSV, DOCS_DIR
from .extract import DEFAULT_MODEL, make_groq_llm
from .policy_agent import COSTS_CSV, SYSTEM_PROMPT as POLICY_PROMPT
from .workflow import Deps, build_graph, ledger_key

RESULTS_CSV = Path(CLAIMS_CSV).parents[1] / "pipeline_results.csv"


def build_sample(claims: pd.DataFrame, costs: pd.DataFrame, per_type: int,
                 clean_n: int, high_bill_n: int = 3) -> pd.DataFrame:
    """Fraud claims of each type, plus clean ones (including legitimately expensive ones).

    Each duplicate is paired with its original, and originals are processed BEFORE the
    copies. That simulates submission order, which the synthetic data does not store.
    """
    typical = dict(zip(costs["procedure"], costs["typical_cost"]))
    clean_all = claims[~claims["is_fraud"]]
    parts = [claims[claims["fraud_type"] == t].head(per_type) for t in FRAUD_TYPES if t != "duplicate_claim"]
    dups = claims[claims["fraud_type"] == "duplicate_claim"].head(per_type)
    originals = {ledger_key(r) for r in dups.to_dict("records")}
    parts.append(clean_all[[ledger_key(r) in originals for r in clean_all.to_dict("records")]])
    parts.append(clean_all.head(clean_n))
    ratio = clean_all["billed_amount"] / clean_all["procedure"].map(typical)
    parts.append(clean_all[ratio > 1.3].head(high_bill_n))
    first = pd.concat(parts).drop_duplicates("claim_id")
    return pd.concat([first, dups])


def run_pipeline_eval(sample: pd.DataFrame, docs_dir: Path, run_fn, sleep: float = 0) -> pd.DataFrame:
    rows, recs = [], sample.to_dict("records")
    for i, rec in enumerate(recs, 1):
        text = (docs_dir / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
        out = run_fn({"claim_id": rec["claim_id"], "doc_text": text})
        rows.append({"claim_id": rec["claim_id"], "is_fraud": bool(rec["is_fraud"]),
                     "fraud_type": rec["fraud_type"], "recommendation": out["recommendation"],
                     "risk_score": out["risk_score"], "signals": ",".join(out.get("signals", {}))})
        print(f"  [{i}/{len(recs)}] {rec['claim_id']} -> {out['recommendation']} ({out['risk_score']})")
        if sleep and i < len(recs):
            time.sleep(sleep)
    return pd.DataFrame(rows)


def report(res: pd.DataFrame) -> dict:
    flagged = res["recommendation"] != "approve"
    fraud = res["is_fraud"]
    tp, fp = int((fraud & flagged).sum()), int((~fraud & flagged).sum())
    fn, tn = int((fraud & ~flagged).sum()), int((~fraud & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print("\n=== Full pipeline vs ground truth ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision: {precision:.1%}   Recall: {recall:.1%}")
    print("\nPer fraud type (flagged / right signal named / high-risk):")
    for t in FRAUD_TYPES:
        sub = res[res["fraud_type"] == t]
        if len(sub):
            named = sum(t in s.split(",") for s in sub["signals"])
            print(f"  {t:30s} n={len(sub):2d}  flagged={int((sub.recommendation != 'approve').sum()):2d}"
                  f"  named={named:2d}  high_risk={int((sub.recommendation == 'high_risk').sum()):2d}")
    wrong = res[~fraud & flagged]
    print(f"\nClean claims wrongly flagged: {len(wrong)}/{int((~fraud).sum())}")
    for r in wrong.to_dict("records"):
        print("  ", r["claim_id"], r["signals"], r["risk_score"])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraud-per-type", type=int, default=4)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=12)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    claims, costs = pd.read_csv(CLAIMS_CSV), pd.read_csv(COSTS_CSV)
    sample = build_sample(claims, costs, args.fraud_per_type, args.clean)
    print(f"Sample: {len(sample)} claims ({int(sample.is_fraud.sum())} fraud)")
    deps = Deps(costs, make_groq_llm(args.model), make_groq_llm(args.model, POLICY_PROMPT))
    graph = build_graph(deps, human_in_loop=False)
    results = run_pipeline_eval(sample, DOCS_DIR, graph.invoke, args.sleep)
    report(results)
    results.to_csv(RESULTS_CSV, index=False)