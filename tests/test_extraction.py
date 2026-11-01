import json
import random

from src.claimguard.data_gen import generate
from src.claimguard.doc_gen import render_document, write_documents
from src.claimguard.extract import FIELDS, parse_json, run_eval


def test_documents_hide_labels_and_keep_facts():
    df = generate(hard=True, save=False)
    rng = random.Random(1)
    for rec in df.head(30).to_dict("records"):
        text = render_document(rec, rng)
        assert rec["policy_id"] in text and rec["hospital_id"] in text
        assert rec["procedure"] in text and rec["diagnosis"] in text
        assert "fraud" not in text.lower()


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_eval_loop_with_perfect_fake_llm(tmp_path):
    df = generate(hard=False, save=False).head(5)
    write_documents(df, tmp_path)
    by_policy = {r["policy_id"]: r for r in df.to_dict("records")}

    def fake_llm(text):   # cheats by looking up the policy number in the document
        pid = next(p for p in by_policy if p in text)
        return json.dumps({f: str(by_policy[pid][f]) for f in FIELDS})

    r = run_eval(df, tmp_path, fake_llm, n=5)
    assert r["parsed"] == 5 and r["exact"] == 5


def test_eval_counts_parse_failures(tmp_path):
    df = generate(hard=False, save=False).head(3)
    write_documents(df, tmp_path)
    r = run_eval(df, tmp_path, lambda t: "I cannot help", n=3)
    assert r["parsed"] == 0 and len(r["mismatches"]) == 3