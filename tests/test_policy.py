import json

import pandas as pd

from src.claimguard.data_gen import generate
from src.claimguard.policy_agent import EXPECTED, evaluate_agent, run_agent
from src.claimguard.policy_rag import BM25, CLAUSES, claim_facts, retrieve_clauses

COSTS = pd.read_csv("data/procedure_costs.csv")


def test_clauses_loaded():
    assert len(CLAUSES) == 10 and "30 days" in CLAUSES["C2"]


def test_bm25_ranks_the_obvious_match_first():
    idx = BM25(["waiting period of thirty days", "sum insured limit", "duplicate claims"])
    assert idx.top_k("admission within the waiting period", 1) == [0]


def test_retrieval_finds_the_expected_clause_for_every_fraud_type():
    df = generate(hard=True, save=False)
    for ftype, expected in EXPECTED.items():
        for rec in df[df.fraud_type == ftype].to_dict("records"):
            got = {c for c, _ in retrieve_clauses(claim_facts(rec, COSTS))}
            assert got & expected


def test_agent_drops_invented_clause_ids():
    rec = generate(hard=False, save=False).iloc[0].to_dict()
    reply = json.dumps({"violations": [{"clause": "C99", "reason": "made up"},
                                       {"clause": "c2", "reason": "waiting period"}]})
    res = run_agent(lambda p: reply, rec, COSTS)
    assert [v["clause"] for v in res["violations"]] == ["C2"]
    assert res["invalid_citations"] == 1


def test_agent_returns_none_on_garbage():
    rec = generate(hard=False, save=False).iloc[0].to_dict()
    assert run_agent(lambda p: "no idea", rec, COSTS) is None


def test_eval_loop_counts_correctly():
    df = generate(hard=False, save=False)

    def oracle(prompt):   # cites C2 only when the waiting-period fact is small
        n = int(prompt.split("Admission was ")[1].split(" days")[0])
        return json.dumps({"violations": [{"clause": "C2", "reason": "x"}] if n <= 30 else []})

    r = evaluate_agent(df, COSTS, oracle, per_type=5, clean=5)
    assert r["stats"]["waiting_period_violation"]["right_clause"] == 5
    assert r["clean_fp"] == 0 and r["parse_failures"] == 0