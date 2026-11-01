"""LLM extraction agent: read a hospital document, return structured fields.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.claimguard.extract --n 15
"""
import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from .doc_gen import CLAIMS_CSV, DOCS_DIR

ROOT = Path(__file__).resolve().parents[2]
MISMATCH_CSV = ROOT / "data" / "extraction_mismatches.csv"
DEFAULT_MODEL = "openai/gpt-oss-20b"
SLEEP_SECONDS = 8   # stays under Groq's free-tier tokens-per-minute limit

SYSTEM_PROMPT = """You extract data from hospital documents.
Return ONLY a JSON object with exactly these keys:
patient_name (string), patient_age (integer), policy_id (string),
policy_start (YYYY-MM-DD), policy_end (YYYY-MM-DD), hospital_id (string),
admission_date (YYYY-MM-DD), discharge_date (YYYY-MM-DD),
diagnosis (string), procedure (string),
billed_amount (number: the TOTAL amount payable, not the advance paid),
sum_insured (number).
Rules:
- Dates in the document are day-first (DD/MM/YYYY). Output them as YYYY-MM-DD.
- Amounts are plain numbers with no currency symbol or commas.
- Copy values exactly as written. Do NOT correct, fix or question anything,
  even if dates look wrong or inconsistent. Your job is extraction only.
- Use null for a field that is missing. Output JSON only, no explanation."""

FIELDS = ["patient_name", "patient_age", "policy_id", "policy_start", "policy_end",
          "hospital_id", "admission_date", "discharge_date", "diagnosis",
          "procedure", "billed_amount", "sum_insured"]
TEXT_FIELDS = {"patient_name", "policy_id", "hospital_id", "diagnosis", "procedure"}
DATE_FIELDS = {"policy_start", "policy_end", "admission_date", "discharge_date"}


class ExtractedClaim(BaseModel):
    patient_name: str
    patient_age: int
    policy_id: str
    policy_start: date
    policy_end: date
    hospital_id: str
    admission_date: date
    discharge_date: date
    diagnosis: str
    procedure: str
    billed_amount: float
    sum_insured: float


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, even if it has fences or chatter."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def extract_claim(llm_fn, doc_text: str):
    """Returns (ExtractedClaim | None, error message | None)."""
    try:
        return ExtractedClaim(**parse_json(llm_fn(doc_text))), None
    except (ValueError, TypeError) as e:
        return None, f"{type(e).__name__}: {str(e)[:150]}"


def make_groq_llm(model: str = DEFAULT_MODEL):
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}

    def call(doc_text: str) -> str:
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, extra_body=extra,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": doc_text}],
                )
                return r.choices[0].message.content or ""
            except Exception as e:   # rate limit or transient error: back off and retry
                if attempt == 3:
                    raise
                print(f"   API error ({type(e).__name__}), waiting 20s...")
                time.sleep(20)
    return call


def field_matches(field: str, expected, got) -> bool:
    if field in TEXT_FIELDS:
        return str(expected).strip().casefold() == str(got).strip().casefold()
    if field in DATE_FIELDS:
        return date.fromisoformat(str(expected)[:10]) == got
    return abs(float(expected) - float(got)) < 1


def run_eval(df: pd.DataFrame, docs_dir: Path, llm_fn, n: int, sleep: float = 0):
    hits = {f: 0 for f in FIELDS}
    parsed = exact = 0
    mismatches = []
    rows = df.head(n).to_dict("records")
    for i, rec in enumerate(rows, 1):
        text = (docs_dir / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
        got, err = extract_claim(llm_fn, text)
        if got is None:
            mismatches.append({"claim_id": rec["claim_id"], "field": "(parse failure)",
                               "expected": "", "got": err})
        else:
            parsed += 1
            ok_all = True
            for f in FIELDS:
                if field_matches(f, rec[f], getattr(got, f)):
                    hits[f] += 1
                else:
                    ok_all = False
                    mismatches.append({"claim_id": rec["claim_id"], "field": f,
                                       "expected": rec[f], "got": getattr(got, f)})
            exact += ok_all
        print(f"  [{i}/{len(rows)}] {rec['claim_id']} {'ok' if got else 'PARSE FAIL'}")
        if sleep and i < len(rows):
            time.sleep(sleep)
    return {"n": len(rows), "parsed": parsed, "exact": exact,
            "field_hits": hits, "mismatches": mismatches}


def print_report(r: dict) -> None:
    n = r["n"]
    print("\n=== Extraction accuracy ===")
    print(f"Claims: {n}   parsed: {r['parsed']}/{n}   fully correct: {r['exact']}/{n} ({r['exact']/n:.0%})")
    print("Per field (of parsed claims):")
    for f, h in r["field_hits"].items():
        print(f"  {f:16s} {h}/{r['parsed']}")
    print(f"Mismatches: {len(r['mismatches'])}")
    for m in r["mismatches"][:10]:
        print("  ", m["claim_id"], m["field"], "| expected:", m["expected"], "| got:", m["got"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    claims = pd.read_csv(CLAIMS_CSV)
    result = run_eval(claims, DOCS_DIR, make_groq_llm(args.model), args.n, sleep=SLEEP_SECONDS)
    print_report(result)
    pd.DataFrame(result["mismatches"]).to_csv(MISMATCH_CSV, index=False)