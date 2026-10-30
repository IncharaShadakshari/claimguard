from src.claimguard.data_gen import generate
from src.claimguard.evaluate import evaluate


def test_baseline_on_easy_data():
    m = evaluate(generate(hard=False, save=False))
    assert m["recall"] >= 0.9
    assert m["precision"] >= 0.8


def test_hard_data_is_actually_harder():
    easy = evaluate(generate(hard=False, save=False))
    hard = evaluate(generate(hard=True, save=False))
    assert hard["recall"] < easy["recall"]
    assert hard["tp"] + hard["fn"] == 75