import pandas as pd

from src.claimguard.ablation import disagreements, in_submission_order, metrics, run_rules_only
from src.claimguard.data_gen import generate

COSTS = pd.read_csv("data/procedure_costs.csv")


def test_rules_only_catches_all_duplicates_even_with_typos():
    df = generate(hard=True, save=False)
    res = run_rules_only(in_submission_order(df), COSTS)
    dups = res[res.fraud_type == "duplicate_claim"]
    assert dups["signals"].str.contains("duplicate_claim").all()


def test_rules_only_never_flags_the_original_of_a_duplicate():
    df = generate(hard=True, save=False)
    res = run_rules_only(in_submission_order(df), COSTS)
    clean_dup_flags = res[~res.is_fraud & res.signals.str.contains("duplicate_claim")]
    assert len(clean_dup_flags) == 0


def test_metrics_and_disagreements():
    df = generate(hard=False, save=False).head(60)
    res = run_rules_only(df, COSTS)
    m = metrics(res)
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == 60
    assert len(disagreements(res, res)) == 0