"""Interactive demo of the human-review step.

Run from the project root (needs GROQ_API_KEY in .env):
    python -m src.claimguard.review_demo --type inflated_billing
"""
import argparse

import pandas as pd
from langgraph.types import Command

from .doc_gen import CLAIMS_CSV, DOCS_DIR
from .extract import DEFAULT_MODEL, make_groq_llm
from .policy_agent import COSTS_CSV, SYSTEM_PROMPT as POLICY_PROMPT
from .workflow import Deps, build_graph

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="date_mismatch")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    claims, costs = pd.read_csv(CLAIMS_CSV), pd.read_csv(COSTS_CSV)
    rec = claims[claims["fraud_type"] == args.type].iloc[0]
    text = (DOCS_DIR / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
    graph = build_graph(Deps(costs, make_groq_llm(args.model), make_groq_llm(args.model, POLICY_PROMPT)))
    cfg = {"configurable": {"thread_id": rec["claim_id"]}}

    print(f"Processing {rec['claim_id']} ...")
    out = graph.invoke({"claim_id": rec["claim_id"], "doc_text": text}, cfg)
    if "__interrupt__" not in out:
        print("Approved automatically:", out["explanation"])
    else:
        info = out["__interrupt__"][0].value
        print("\n--- PAUSED FOR HUMAN REVIEW ---")
        print(info["explanation"])
        choice = input("\nYour decision: [a] approve  [i] send for investigation > ").strip().lower()
        action = "approve" if choice == "a" else "investigate"
        out = graph.invoke(Command(resume={"action": action}), cfg)
        print("\nFinal decision recorded:", out["final_decision"])