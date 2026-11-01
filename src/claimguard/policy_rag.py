"""Policy retrieval: a sample health policy, a small BM25 search, and claim facts.

The policy text is a fictional sample written in common health-insurance wording.
It is NOT a real insurer's policy.
"""
import math
import re
from collections import Counter

import pandas as pd

POLICY_TEXT = """\
C1. Policy period. A claim is payable only if the hospitalisation, from admission to discharge, falls within the policy period stated in the policy schedule. Admission after the policy period has ended is not payable.
C2. Initial waiting period. No claim is payable for any illness or treatment where admission occurs within 30 days from the policy start date. This waiting period does not apply to hospitalisation caused by accidental injury.
C3. Accidental injury. Hospitalisation for fractures and other injuries caused by an accident is covered from the first day of the policy, subject to the sum insured.
C4. Medical necessity. The procedure performed must be consistent with the diagnosis. A procedure that is not an accepted treatment for the stated diagnosis is not payable and is referred for medical review.
C5. Reasonable and customary charges. Charges are payable up to the usual charge for the procedure. A billed amount more than 1.5 times the usual charge for the procedure is referred for review.
C6. Sum insured. The total amount payable in a policy year cannot exceed the sum insured. A billed amount above the sum insured is payable only up to the sum insured.
C7. Document consistency. The discharge summary and final bill must show consistent dates. The discharge date cannot be earlier than the admission date. Inconsistent documents are referred for investigation.
C8. Duplicate claims. A claim for the same hospitalisation cannot be submitted twice. The second submission is not payable.
C9. Exclusions. Cosmetic surgery, infertility treatment, self-inflicted injury and treatment outside India are excluded.
C10. Cashless pre-authorisation. Planned treatment at a network hospital requires pre-authorisation before admission. Emergency admissions must be intimated within 48 hours.
"""

STOPWORDS = {"a", "an", "and", "the", "of", "to", "is", "are", "was", "for", "in", "or", "be",
             "by", "at", "on", "as", "not", "it", "this", "that", "with", "from", "any", "than"}


def load_clauses(text: str = POLICY_TEXT) -> dict:
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^(C\d+)\.\s*(.+)$", text, flags=re.M)}


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    """Tiny BM25 ranker, no external dependencies."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        n = len(self.docs)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top_k(self, query: str, k: int) -> list:
        q = tokenize(query)
        scores = []
        for d, length in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[:k] if scores[i] > 0]


CLAUSES = load_clauses()
_IDS = list(CLAUSES)
_INDEX = BM25([CLAUSES[i] for i in _IDS])


def claim_facts(c: dict, costs: pd.DataFrame) -> list:
    """Computed facts about one claim, as plain sentences. Dates are calculated in code
    (language models are unreliable at date arithmetic)."""
    typical = dict(zip(costs["procedure"], costs["typical_cost"]))
    usual_dx = dict(zip(costs["procedure"], costs["diagnosis"]))
    ps, pe = pd.Timestamp(c["policy_start"]), pd.Timestamp(c["policy_end"])
    adm, dis = pd.Timestamp(c["admission_date"]), pd.Timestamp(c["discharge_date"])
    days = (adm - ps).days
    ratio = c["billed_amount"] / typical[c["procedure"]]
    inside = ps <= adm <= pe
    return [
        f"Admission was {days} days after the policy start date.",
        f"Admission was {'inside' if inside else 'outside'} the policy period.",
        f"The discharge date is {'before' if dis < adm else 'after'} the admission date.",
        f"The billed amount is {ratio:.1f} times the usual charge for the procedure.",
        f"The billed amount {'exceeds' if c['billed_amount'] > c['sum_insured'] else 'is within'} the sum insured.",
        f"The procedure {c['procedure']} was billed for the diagnosis {c['diagnosis']}; "
        f"the usual diagnosis for this procedure is {usual_dx[c['procedure']]}.",
    ]


def retrieve_clauses(queries: list, per_query: int = 1) -> list:
    """Multi-query retrieval: one search per fact sentence, results merged in order."""
    found = []
    for q in queries:
        for i in _INDEX.top_k(q, per_query):
            if _IDS[i] not in found:
                found.append(_IDS[i])
    return [(cid, CLAUSES[cid]) for cid in found]