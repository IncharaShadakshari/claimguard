from src.claimguard.data_gen import generate
from src.claimguard.evaluate import evaluate


def test_baseline_beats_chance():
    m = evaluate(generate())
    assert m["recall"] >= 0.9
    assert m["precision"] >= 0.8
    assert m["tp"] + m["fn"] == 75