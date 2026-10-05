
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI(
    title="Bank Loan Default API"
)
model = joblib.load("rf_bankloan.pkl")
class LoanApplication(BaseModel):
    AGE: int
    EMPLOY: int
    ADDRESS: int
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float
@app.get("/")
def home():
    return {
        "message": "Bank Loan Probability API"
    }





@app.post("/predict")
def predict(data: LoanApplication):
    features = pd.DataFrame([{
        "AGE": data.AGE,
        "EMPLOY": data.EMPLOY,
        "ADDRESS": data.ADDRESS,
        "DEBTINC": data.DEBTINC,
        "CREDDEBT": data.CREDDEBT,
        "OTHDEBT": data.OTHDEBT
    }])
    probability = model.predict_proba(features)[0][1]
    return {
        "Probability_of_Default":
        round(float(probability),4)
    }
