"""Turn structured claims into messy, realistic hospital documents (plain text).

Run from the project root (after data_gen):
    python -m src.claimguard.doc_gen
"""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_CSV = ROOT / "data" / "synthetic" / "claims.csv"
DOCS_DIR = ROOT / "data" / "synthetic" / "docs"
SEED = 7

HOSPITAL_NAMES = [
    "Sri Sai Multispeciality Hospital", "City General Hospital", "Lakshmi Nursing Home",
    "Apex Care Hospital", "Green Valley Medical Centre", "Sunrise Hospital",
    "Mahaveer Hospital", "Metro Heart Institute", "Kaveri Hospital", "Unity Health Centre",
]


def fmt_date(d, style):
    d = pd.Timestamp(d)
    return d.strftime(["%d/%m/%Y", "%d %b %Y", "%Y-%m-%d", "%d-%m-%Y"][style])


def fmt_money(x, style):
    return [f"Rs. {x:,.0f}/-", f"\u20b9{x:,.2f}", f"INR {x:.0f}"][style]


def render_document(c: dict, rng: random.Random) -> str:
    """c is one claim as a dict. Ground-truth labels are never written into the text."""
    ds, ms = rng.randrange(4), rng.randrange(3)
    name = c["patient_name"].upper() if rng.random() < 0.4 else c["patient_name"]
    age = f"{c['patient_age']} yrs" if rng.random() < 0.5 else f"{c['patient_age']}Y"
    hosp = HOSPITAL_NAMES[int(c["hospital_id"][1:]) % len(HOSPITAL_NAMES)]
    advance = round(c["billed_amount"] * 0.2, -2)
    return f"""{hosp}
Hospital Reg. No: {c['hospital_id']}

DISCHARGE SUMMARY
Patient name : {name}
Age          : {age}
Policy No    : {c['policy_id']}
Policy period: {fmt_date(c['policy_start'], ds)} to {fmt_date(c['policy_end'], ds)}
Sum insured  : {fmt_money(c['sum_insured'], ms)}
Date of admission : {fmt_date(c['admission_date'], ds)}
Date of discharge : {fmt_date(c['discharge_date'], ds)}
Diagnosis    : {c['diagnosis']}
Procedure    : {c['procedure']}
Condition at discharge: Stable. Review after 7 days.

------------------------------------------------
FINAL BILL   (Bill no: B{rng.randint(10000, 99999)})
Patient: {name}
Advance paid           : {fmt_money(advance, ms)}
Total amount payable   : {fmt_money(c['billed_amount'], ms)}
Authorised signatory: ____________
"""


def write_documents(df: pd.DataFrame, out_dir: Path = DOCS_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for rec in df.to_dict("records"):
        text = render_document(rec, rng)
        (out_dir / f"{rec['claim_id']}.txt").write_text(text, encoding="utf-8")
    return len(df)


if __name__ == "__main__":
    n = write_documents(pd.read_csv(CLAIMS_CSV))
    print(f"Wrote {n} documents to {DOCS_DIR}")