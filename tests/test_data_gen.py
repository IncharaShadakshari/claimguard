import pandas as pd

from src.claimguard.data_gen import FRAUD_TYPES, WAITING_DAYS, generate


def test_counts_and_labels():
    df = generate(save=False)
    assert len(df) == 500
    assert df["is_fraud"].sum() == 75
    assert set(df.loc[df.is_fraud, "fraud_type"]) == set(FRAUD_TYPES)
    assert df.claim_id.is_unique


def test_normal_claims_are_clean():
    df = generate(save=False)
    ok = df[~df.is_fraud].copy()
    for col in ["policy_start", "policy_end", "admission_date", "discharge_date"]:
        ok[col] = pd.to_datetime(ok[col])
    assert (ok.discharge_date >= ok.admission_date).all()
    assert (ok.admission_date >= ok.policy_start).all()
    assert (ok.admission_date <= ok.policy_end).all()
    assert ((ok.admission_date - ok.policy_start).dt.days > WAITING_DAYS).all()