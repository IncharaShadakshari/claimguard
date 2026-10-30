"""Rule-based fraud checks. No LLM, no ML: this is the baseline the agents must beat."""
import pandas as pd

INFLATION_FACTOR = 1.5   # flag bills above 1.5x the typical cost (normal bills max out at 1.2x)
WAITING_DAYS = 30        # same assumption as the data generator
DUPLICATE_KEY = ["patient_name", "policy_id", "hospital_id", "procedure", "admission_date"]
RULES = [
    "duplicate_claim",
    "inflated_billing",
    "date_mismatch",
    "diagnosis_procedure_mismatch",
    "waiting_period_violation",
]


def check_claims(claims: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Return the claims with one True/False column per rule, plus `flagged` and `flag_reasons`."""
    df = claims.copy()
    for col in ["policy_start", "policy_end", "admission_date", "discharge_date"]:
        df[col] = pd.to_datetime(df[col])

    typical_cost = df["procedure"].map(dict(zip(costs["procedure"], costs["typical_cost"])))
    expected_dx = df["procedure"].map(dict(zip(costs["procedure"], costs["diagnosis"])))
    days_since_start = (df["admission_date"] - df["policy_start"]).dt.days
    is_accident_type = df["procedure"].str.contains("ORIF")   # fractures: no waiting period

    flags = pd.DataFrame(index=df.index)
    # keep=False flags BOTH copies, because without a submission timestamp we cannot tell
    # which one came first. The original therefore becomes a known false positive.
    flags["duplicate_claim"] = df.duplicated(subset=DUPLICATE_KEY, keep=False)
    flags["inflated_billing"] = df["billed_amount"] > INFLATION_FACTOR * typical_cost
    flags["date_mismatch"] = (
        (df["discharge_date"] < df["admission_date"])
        | (df["admission_date"] < df["policy_start"])
        | (df["admission_date"] > df["policy_end"])
    )
    flags["diagnosis_procedure_mismatch"] = df["diagnosis"] != expected_dx
    flags["waiting_period_violation"] = (
        (days_since_start >= 0) & (days_since_start <= WAITING_DAYS) & ~is_accident_type
    )

    out = pd.concat([df, flags], axis=1)
    out["flagged"] = flags.any(axis=1)
    out["flag_reasons"] = flags.apply(lambda r: ",".join(r.index[r]), axis=1)
    return out