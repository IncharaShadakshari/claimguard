"""The claim triage workflow: extract -> rule checks -> policy agent -> decision -> human review.

All the logic lives in plain functions (make_nodes). build_graph() only wires them into a
LangGraph graph, and run_sequential() runs the same functions without LangGraph (for tests).
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, TypedDict

import pandas as pd

from .extract import extract_claim
from .policy_agent import run_agent
from .rules import RULES, check_claims

# How strongly each signal suggests fraud. Hand-set starting values, not learned.
SIGNAL_WEIGHTS = {
    "date_mismatch": 0.9,
    "duplicate_claim": 0.9,
    "waiting_period_violation": 0.7,
    "diagnosis_procedure_mismatch": 0.7,
    "inflated_billing": 0.5,        # lower: legitimate complications can look like inflation
    "extraction_failed": 0.5,       # unreadable document: a person should look at it
}
CLAUSE_TO_SIGNAL = {
    "C1": "date_mismatch", "C7": "date_mismatch", "C2": "waiting_period_violation",
    "C4": "diagnosis_procedure_mismatch", "C5": "inflated_billing", "C8": "duplicate_claim",
}
APPROVE_BELOW = 0.3
HIGH_RISK_FROM = 0.8


class ClaimState(TypedDict, total=False):
    claim_id: str
    doc_text: str
    extracted: dict
    extraction_error: str
    rule_signals: list
    policy_findings: list
    policy_error: str
    signals: dict
    risk_score: float
    recommendation: str      # approve | human_review | high_risk
    explanation: str
    human_decision: dict
    final_decision: str


@dataclass
class Deps:
    costs: pd.DataFrame
    extract_llm: Callable
    policy_llm: Callable
    ledger: dict = field(default_factory=dict)   # duplicate key -> first claim_id seen


def ledger_key(c: dict) -> tuple:
    return (c["policy_id"], c["hospital_id"], c["procedure"], str(c["admission_date"]))


def score_signals(signals: dict) -> tuple:
    """Noisy-OR: each signal independently adds risk. Returns (risk, recommendation)."""
    keep = 1.0
    for s in signals:
        keep *= 1 - SIGNAL_WEIGHTS[s]
    risk = round(1 - keep, 2)
    if risk < APPROVE_BELOW:
        return risk, "approve"
    return risk, ("high_risk" if risk >= HIGH_RISK_FROM else "human_review")


def make_nodes(deps: Deps) -> dict:
    def extract_node(state):
        got, err = extract_claim(deps.extract_llm, state["doc_text"])
        if got is None:
            return {"extraction_error": err}
        clean = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in got.model_dump().items()}
        return {"extracted": clean}

    def rules_node(state):
        c = state["extracted"]
        row = check_claims(pd.DataFrame([c]), deps.costs).iloc[0]
        found = [r for r in RULES if r != "duplicate_claim" and bool(row[r])]
        key = ledger_key(c)
        if key in deps.ledger and deps.ledger[key] != state["claim_id"]:
            found.append("duplicate_claim")      # an earlier claim has the same key
        else:
            deps.ledger[key] = state["claim_id"]
        return {"rule_signals": found}

    def policy_node(state):
        res = run_agent(deps.policy_llm, state["extracted"], deps.costs)
        if res is None:
            return {"policy_findings": [], "policy_error": "unusable policy-agent reply"}
        return {"policy_findings": res["violations"]}

    def decide_node(state):
        signals = {}
        for s in state.get("rule_signals", []):
            signals.setdefault(s, []).append("rule check")
        for v in state.get("policy_findings", []):
            s = CLAUSE_TO_SIGNAL.get(v["clause"])
            if s:
                signals.setdefault(s, []).append(f"policy {v['clause']}: {v['reason']}")
        if state.get("extraction_error"):
            signals["extraction_failed"] = [state["extraction_error"]]
        risk, rec = score_signals(signals)
        if signals:
            lines = [f"- {s}: " + "; ".join(ev) for s, ev in signals.items()]
        else:
            lines = ["- no fraud signals found"]
        text = (f"Risk score {risk:.2f} -> {rec}.\n" + "\n".join(lines) +
                "\nThis is a recommendation only. A human makes the final decision on any flagged claim.")
        out = {"signals": signals, "risk_score": risk, "recommendation": rec, "explanation": text}
        if rec == "approve":
            out["final_decision"] = "approve"
        return out

    def human_review_node(state):
        from langgraph.types import interrupt
        answer = interrupt({
            "claim_id": state["claim_id"],
            "recommendation": state["recommendation"],
            "risk_score": state["risk_score"],
            "explanation": state["explanation"],
            "extracted": state.get("extracted"),
        })
        return {"human_decision": answer, "final_decision": answer.get("action", "unspecified")}

    return {"extract": extract_node, "rules": rules_node, "policy": policy_node,
            "decide": decide_node, "human_review": human_review_node}


def run_sequential(deps: Deps, state: dict) -> dict:
    """Same steps as the graph, without LangGraph. Used by tests."""
    nodes = make_nodes(deps)
    state = dict(state)
    state.update(nodes["extract"](state))
    if not state.get("extraction_error"):
        state.update(nodes["rules"](state))
        state.update(nodes["policy"](state))
    state.update(nodes["decide"](state))
    return state


def build_graph(deps: Deps, human_in_loop: bool = True):
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(deps)
    g = StateGraph(ClaimState)
    for name in ("extract", "rules", "policy", "decide"):
        g.add_node(name, nodes[name])
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract",
                            lambda s: "decide" if s.get("extraction_error") else "rules",
                            {"rules": "rules", "decide": "decide"})
    g.add_edge("rules", "policy")
    g.add_edge("policy", "decide")
    if not human_in_loop:
        g.add_edge("decide", END)
        return g.compile()
    from langgraph.checkpoint.memory import InMemorySaver
    g.add_node("human_review", nodes["human_review"])
    g.add_conditional_edges("decide",
                            lambda s: END if s["recommendation"] == "approve" else "human_review",
                            {END: END, "human_review": "human_review"})
    g.add_edge("human_review", END)
    return g.compile(checkpointer=InMemorySaver())