"""Measure the rule baseline against the ground-truth fraud labels.

Run from the project root:
    python -m src.claimguard.evaluate
"""
from pathlib import Path

import pandas as pd

from .rules import RULES, check_claims

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_CSV = ROOT / "data" / "synthetic" / "claims.csv"
COSTS_CSV = ROOT / "data" / "procedure_costs.csv"


def evaluate(claims: pd.DataFrame | None = None) -> dict:
    if claims is None:
        claims = pd.read_csv(CLAIMS_CSV)
    costs = pd.read_csv(COSTS_CSV)
    res = check_claims(claims, costs)

    fraud, flagged = res["is_fraud"].astype(bool), res["flagged"]
    tp = int((fraud & flagged).sum())
    fp = int((~fraud & flagged).sum())
    fn = int((fraud & ~flagged).sum())
    tn = int((~fraud & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_type = {}
    for rule in RULES:
        subset = res[res["fraud_type"] == rule]
        per_type[rule] = {
            "n": len(subset),
            "caught": int(subset["flagged"].sum()),
            "caught_by_correct_rule": int(subset[rule].sum()),
        }

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "per_type": per_type,
        "false_positives": res[~fraud & flagged][["claim_id", "flag_reasons"]],
        "missed_fraud": res[fraud & ~flagged][["claim_id", "fraud_type"]],
    }


def print_report(m: dict) -> None:
    print("=== Rule baseline vs ground truth ===")
    print(f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"Precision: {m['precision']:.1%}")
    print(f"Recall:    {m['recall']:.1%}")
    print(f"F1:        {m['f1']:.1%}")
    print(f"False positive rate: {m['false_positive_rate']:.1%}")
    print("\nPer fraud type:")
    for name, s in m["per_type"].items():
        print(f"  {name:32s} n={s['n']:3d}  caught={s['caught']:3d}  by correct rule={s['caught_by_correct_rule']:3d}")
    fps = m["false_positives"]["flag_reasons"].value_counts()
    print(f"\nFalse positives ({len(m['false_positives'])}):")
    print(fps.to_string() if len(fps) else "  none")
    missed = m["missed_fraud"]["fraud_type"].value_counts()
    print(f"\nMissed fraud ({len(m['missed_fraud'])}):")
    print(missed.to_string() if len(missed) else "  none")


if __name__ == "__main__":
    print_report(evaluate())