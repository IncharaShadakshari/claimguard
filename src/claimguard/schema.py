from datetime import date
from pydantic import BaseModel

class Claim(BaseModel):
    claim_id: str
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
    # ground truth, used only for evaluation, never as a model input
    is_fraud: bool = False
    fraud_type: str | None = None