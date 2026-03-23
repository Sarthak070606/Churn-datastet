from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

class CustomerData(BaseModel):
    CreditScore: float
    Geography_Germany: int
    Geography_Spain: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    
    features = np.array([[
        data.CreditScore,
        data.Geography_Germany,
        data.Geography_Spain,
        data.Gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary
    ]])
    
    features = scaler.transform(features)
    
    probability = model.predict_proba(features)[0][1]
    
    threshold = 0.3
    prediction = 1 if probability > threshold else 0
    
    message = "Customer likely to churn" if prediction == 1 else "Customer not likely to churn"
    
    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "message": message
    }