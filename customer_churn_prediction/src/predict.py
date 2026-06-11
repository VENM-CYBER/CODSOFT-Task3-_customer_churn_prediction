import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "customer_churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

sample = pd.DataFrame({
    "CreditScore": [650],
    "Geography": ["France"],
    "Gender": ["Male"],
    "Age": [40],
    "Tenure": [5],
    "Balance": [60000],
    "NumOfProducts": [2],
    "HasCrCard": [1],
    "IsActiveMember": [1],
    "EstimatedSalary": [50000]
})

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", prediction)
print("Churn Probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Customer likely to leave")
else:
    print("Customer likely to stay")