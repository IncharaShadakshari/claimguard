import json
import re

import pandas as pd
import pytest

from src.claimguard.data_gen import generate
from src.claimguard.doc_gen import write_documents
from src.claimguard.extract import FIELDS
from src.claimguard.workflow import Deps, build_graph, ledger_key, run_sequential, score_signals

COSTS = pd.read_csv("data/procedure_costs.csv")
DX = dict(zip(COSTS["procedure"], COSTS["diagnosis"]))


def fake_extract_for(df):
    by_policy = {}
    for r in df.to_dict("records"):
        by_policy.setdefault(r["policy_id"], r)

    def fake(text):
        pid = next(p for p in by_policy if p in text)
        return json.dumps({f: str(by_policy[pid][f]) for f in FIELDS})
    return fake


def fake_policy(prompt):
    """Stands in for the LLM: applies the clause thresholds to the fact sentences."""
    v = []
    days = int(re.search(r"Admission was (\d+) days", prompt).group(1))
    if days <= 30:
        v.append({"clause": "C2", "reason": "within waiting period"})
    if "outside the policy period" in prompt:
        v.append({"clause": "C1", "reason": "outside policy period"})
    if "is before the admission date" in prompt:
        v.append({"clause": "C7", "reason": "discharge before admission"})
    if float(re.search(r"is ([\d.]+) times", prompt).group(1)) > 1.5:
        v.append({"clause": "C5", "reason": "above 1.5x usual charge"})
    m = re.search(r"billed for the diagnosis (.+?); the usual diagnosis for this procedure is (.+?)\.", prompt)
    if m and m.group(1) != m.group(2):
        v.append({"clause": "C4", "reason": "diagnosis does not fit procedure"})
    return json.dumps({"violations": v})


def setup(df, tmp_path):
    write_documents(df, tmp_path)
    return fake_extract_for(df)


def run_one(deps, rec, tmp_path):
    text = (tmp_path / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
    return run_sequential(deps, {"claim_id": rec["claim_id"], "doc_text": text})


def test_scoring_thresholds():
    assert score_signals({}) == (0.0, "approve")
    assert score_signals({"inflated_billing": []})[1] == "human_review"
    assert score_signals({"date_mismatch": []})[1] == "high_risk"
    assert score_signals({"waiting_period_violation": [], "inflated_billing": []})[1] == "high_risk"


def test_each_fraud_type_is_flagged_and_clean_claims_pass(tmp_path):
    df = generate(hard=False, save=False)
    ext = setup(df, tmp_path)
    deps = Deps(COSTS, ext, fake_policy)
    for ftype in ["inflated_billing", "date_mismatch", "diagnosis_procedure_mismatch",
                  "waiting_period_violation"]:
        for rec in df[df.fraud_type == ftype].head(3).to_dict("records"):
            out = run_one(deps, rec, tmp_path)
            assert out["recommendation"] != "approve", ftype
            assert ftype in out["signals"], (ftype, out["signals"])
    for rec in df[~df.is_fraud].head(8).to_dict("records"):
        assert run_one(Deps(COSTS, ext, fake_policy), rec, tmp_path)["recommendation"] == "approve"


def test_duplicate_is_flagged_only_on_second_arrival(tmp_path):
    df = generate(hard=True, save=False)
    ext = setup(df, tmp_path)
    copy = df[df.fraud_type == "duplicate_claim"].iloc[0].to_dict()
    orig = next(r for r in df[~df.is_fraud].to_dict("records") if ledger_key(r) == ledger_key(copy))
    deps = Deps(COSTS, ext, fake_policy)
    first = run_one(deps, orig, tmp_path)
    second = run_one(deps, copy, tmp_path)
    assert "duplicate_claim" not in first["signals"]
    assert "duplicate_claim" in second["signals"] and second["recommendation"] == "high_risk"


def test_unreadable_document_goes_to_a_human(tmp_path):
    df = generate(hard=False, save=False).head(3)
    write_documents(df, tmp_path)
    out = run_one(Deps(COSTS, lambda t: "garbage", fake_policy), df.iloc[0].to_dict(), tmp_path)
    assert out["recommendation"] == "human_review" and "extraction_failed" in out["signals"]


def test_graph_gives_same_answers_as_sequential(tmp_path):
    pytest.importorskip("langgraph")
    df = generate(hard=False, save=False)
    ext = setup(df, tmp_path)
    graph = build_graph(Deps(COSTS, ext, fake_policy), human_in_loop=False)
    for ftype in ["date_mismatch", "inflated_billing"]:
        rec = df[df.fraud_type == ftype].iloc[0].to_dict()
        text = (tmp_path / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
        seq = run_one(Deps(COSTS, ext, fake_policy), rec, tmp_path)
        out = graph.invoke({"claim_id": rec["claim_id"], "doc_text": text})
        assert out["recommendation"] == seq["recommendation"]
        assert out["risk_score"] == seq["risk_score"]


def test_graph_pauses_for_human_and_resumes(tmp_path):
    pytest.importorskip("langgraph")
    from langgraph.types import Command
    df = generate(hard=False, save=False)
    ext = setup(df, tmp_path)
    graph = build_graph(Deps(COSTS, ext, fake_policy), human_in_loop=True)
    rec = df[df.fraud_type == "date_mismatch"].iloc[0].to_dict()
    text = (tmp_path / f"{rec['claim_id']}.txt").read_text(encoding="utf-8")
    cfg = {"configurable": {"thread_id": rec["claim_id"]}}
    paused = graph.invoke({"claim_id": rec["claim_id"], "doc_text": text}, cfg)
    assert "__interrupt__" in paused
    done = graph.invoke(Command(resume={"action": "investigate"}), cfg)
    assert done["final_decision"] == "investigate"


def test_pipeline_eval_end_to_end_with_fake_llms(tmp_path):
    from src.claimguard.pipeline_eval import build_sample, report, run_pipeline_eval
    df = generate(hard=True, save=False)
    ext = setup(df, tmp_path)
    sample = build_sample(df, COSTS, per_type=3, clean_n=5)
    deps = Deps(COSTS, ext, fake_policy)
    res = run_pipeline_eval(sample, tmp_path, lambda s: run_sequential(deps, s))
    # every duplicate is processed after its original
    ids = list(res["claim_id"])
    for d in sample[sample.fraud_type == "duplicate_claim"].to_dict("records"):
        orig = next(r for r in df[~df.is_fraud].to_dict("records") if ledger_key(r) == ledger_key(d))
        assert ids.index(orig["claim_id"]) < ids.index(d["claim_id"])
    m = report(res)
    assert m["recall"] >= 0.9