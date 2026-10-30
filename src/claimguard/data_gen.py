"""Generate synthetic health insurance claims with injected, labeled fraud.

Run from the project root:
    python -m src.claimguard.data_gen
"""
import copy
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

from .schema import Claim

ROOT = Path(__file__).resolve().parents[2]
COSTS_CSV = ROOT / "data" / "procedure_costs.csv"
OUT_CSV = ROOT / "data" / "synthetic" / "claims.csv"            # hard version
OUT_CSV_EASY = ROOT / "data" / "synthetic" / "claims_easy.csv"  # easy version

SEED = 42
N_CLAIMS = 500
FRAUD_PER_TYPE = 15          # 5 types x 15 = 75 fraud claims = 15% of 500
WAITING_DAYS = 30            # initial waiting period assumed for every policy
FRAUD_TYPES = [
    "duplicate_claim",
    "inflated_billing",
    "date_mismatch",
    "diagnosis_procedure_mismatch",
    "waiting_period_violation",
]
AGE_RANGE = {
    "Normal delivery": (20, 38),
    "Caesarean section": (20, 40),
    "Hysterectomy": (35, 55),
    "Total knee replacement": (50, 80),
    "Cataract surgery": (50, 85),
}

fake = Faker("en_IN")


def make_normal(row, costs, hard=False):
    """One clean claim: valid dates, after waiting period, bill near typical cost.

    hard=True: 5% of legitimate claims get a higher bill (complications), which
    looks like inflation but is not fraud.
    """
    start = date(2023, 1, 1) + timedelta(days=random.randint(0, 900))
    end = start + timedelta(days=364)
    admit = start + timedelta(days=random.randint(WAITING_DAYS + 1, 330))
    stay = 1 if "dialysis" in row.procedure.lower() else random.randint(1, 7)
    low, high = AGE_RANGE.get(row.procedure, (18, 80))
    billed = round(costs[row.procedure] * random.uniform(0.8, 1.2), -2)
    if hard and random.random() < 0.05:
        billed = round(billed * random.uniform(1.3, 1.8), -2)
    return {
        "claim_id": "",
        "patient_name": fake.name(),
        "patient_age": random.randint(low, high),
        "policy_id": f"POL{random.randint(100000, 999999)}",
        "policy_start": start,
        "policy_end": end,
        "hospital_id": f"H{random.randint(1, 20):03d}",
        "admission_date": admit,
        "discharge_date": admit + timedelta(days=stay),
        "diagnosis": row.diagnosis,
        "procedure": row.procedure,
        "billed_amount": billed,
        "sum_insured": random.choice([300_000, 500_000, 1_000_000]),
        "is_fraud": False,
        "fraud_type": None,
    }


def label(d, fraud_type):
    d["is_fraud"] = True
    d["fraud_type"] = fraud_type
    return d


def typo(name):
    """Drop one letter from the name, like a sloppy resubmission."""
    pos = random.randint(1, len(name) - 2)
    return name[:pos] + name[pos + 1:]


def fraud_duplicate(source, hard=False):
    """Same patient, hospital, dates and bill submitted again.

    hard=True: half of the resubmissions have a typo in the patient name.
    """
    d = copy.deepcopy(source)
    if hard and random.random() < 0.5:
        d["patient_name"] = typo(d["patient_name"])
    return label(d, "duplicate_claim")


def fraud_inflated(base, costs, hard=False):
    low, high = (1.3, 2.5) if hard else (2, 4)   # hard: subtler inflation
    base["billed_amount"] = round(costs[base["procedure"]] * random.uniform(low, high), -2)
    base["sum_insured"] = 1_000_000
    return label(base, "inflated_billing")


def fraud_date_mismatch(base):
    stay = (base["discharge_date"] - base["admission_date"]).days
    if random.random() < 0.5:   # admission after the policy has expired
        base["admission_date"] = base["policy_end"] + timedelta(days=random.randint(5, 60))
        base["discharge_date"] = base["admission_date"] + timedelta(days=stay)
    else:                       # discharge before admission
        base["discharge_date"] = base["admission_date"] - timedelta(days=random.randint(1, 5))
    return label(base, "date_mismatch")


def fraud_dx_mismatch(base, rows):
    other = random.choice([r for r in rows if r.diagnosis != base["diagnosis"]])
    base["diagnosis"] = other.diagnosis      # procedure stays the same
    return label(base, "diagnosis_procedure_mismatch")


def fraud_waiting(base):
    stay = (base["discharge_date"] - base["admission_date"]).days
    base["admission_date"] = base["policy_start"] + timedelta(days=random.randint(1, 20))
    base["discharge_date"] = base["admission_date"] + timedelta(days=stay)
    return label(base, "waiting_period_violation")


def generate(hard=True, save=True):
    random.seed(SEED)
    Faker.seed(SEED)

    df = pd.read_csv(COSTS_CSV)
    costs = dict(zip(df["procedure"], df["typical_cost"]))
    rows = list(df.itertuples(index=False))

    n_fraud = FRAUD_PER_TYPE * len(FRAUD_TYPES)
    normals = [make_normal(random.choice(rows), costs, hard) for _ in range(N_CLAIMS - n_fraud)]
    claims = list(normals)

    for src in random.sample(normals, FRAUD_PER_TYPE):
        claims.append(fraud_duplicate(src, hard))

    def fresh():
        return make_normal(random.choice(rows), costs, hard)

    for _ in range(FRAUD_PER_TYPE):
        claims.append(fraud_inflated(fresh(), costs, hard))
        claims.append(fraud_date_mismatch(fresh()))
        claims.append(fraud_dx_mismatch(fresh(), rows))
        # accident-type procedures are exempt from waiting periods
        base = make_normal(random.choice([r for r in rows if "ORIF" not in r.procedure]), costs, hard)
        claims.append(fraud_waiting(base))

    random.shuffle(claims)
    for i, c in enumerate(claims, 1):
        c["claim_id"] = f"CLM{i:05d}"

    validated = [Claim(**c) for c in claims]   # schema check
    out = pd.DataFrame([c.model_dump() for c in validated])
    if save:
        path = OUT_CSV if hard else OUT_CSV_EASY
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    for hard in (False, True):
        out = generate(hard=hard)
        name = "hard" if hard else "easy"
        print(f"[{name}] saved {len(out)} claims, fraud rate {out['is_fraud'].mean():.1%}")