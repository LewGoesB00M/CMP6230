from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

### UPDATED IMPROVEMENT AFTER ORIGINAL PIPELINE ###
# Originally, the user had to input 23 columns of data due to one-hot encoding.
# However, we can reduce this to 12 columns by performing the encoding and scaling as part of this script.
from sklearn.preprocessing import LabelEncoder, StandardScaler 
###################################################

# Initialise the FastAPI object so that the endpoints can be accessed.
app = FastAPI()

# Loading our model from a file.
path = "dags/mlartifacts/0/5689bb2e89764b2facf53e4d032c2b56/artifacts/model/model.pkl"
pickle_input = open(path, "rb")
mdl = pickle.load(pickle_input)

# Also loading the LabelEncoders and StandardScaler used in training the model.
## Education LabelEncoder
path = "/home/lewis/CMP6230/le_education.pkl"
pickle_input = open(path, "rb")
le_education = pickle.load(pickle_input)

## Previous Defaults LabelEncoder
path = "/home/lewis/CMP6230/le_previousDefaults.pkl"
pickle_input = open(path, "rb")
le_defaults = pickle.load(pickle_input)

## StandardScaler
path = "/home/lewis/CMP6230/standard_scaler.pkl"
pickle_input = open(path, "rb")
sc = pickle.load(pickle_input)

### UPDATED AFTER ORIGINAL PIPELINE
# Now only needs 13 column inputs instead of 23.
class LoanInput(BaseModel):
    person_age: int
    person_gender: str
    person_education: str
    person_income: int
    person_emp_exp: int 
    person_home_ownership: str
    loan_amnt: int
    loan_intent: str
    loan_int_rate: float 
    loan_percent_income: float 
    cb_person_cred_hist_length: int
    credit_score: int
    previous_loan_defaults_on_file: str

class LoanPrediction(BaseModel):
    person_age: int
    person_education: int
    person_income: int
    person_emp_exp: int 
    loan_amnt: int
    loan_int_rate: float 
    loan_percent_income: float 
    cb_person_cred_hist_length: int
    credit_score: int
    previous_loan_defaults_on_file: int
    loan_status: int # Include the target in the prediction
    person_gender_female: int 
    person_gender_male: int 
    person_home_ownership_MORTGAGE: int 
    person_home_ownership_OTHER: int 
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int 
    loan_intent_DEBTCONSOLIDATION: int 
    loan_intent_EDUCATION: int 
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int 
    loan_intent_PERSONAL: int 
    loan_intent_VENTURE: int
   

# If a GET request is performed to the main page at 127.0.0.1:8000,
# the "Hello" key with the value "World" will be returned.   
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Retrieves a given row of the dataset.
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# The model serving endpoint to POST data to and receive a prediction from.    
@app.post("/loan/predict/single")
def predict_loan_single(data: LoanInput):
    # Convert the LoanInput object entered by the user into a dict.
    data_dict = data.dict()
    
    # Make the dict into a DataFrame so that it can be encoded and scaled.
    df = pd.DataFrame([data_dict])
    
    # Using the two LabelEncoders saved when the model was trained,
    # apply label encoding to the label encoded columns.
    df["person_education"] = le_education.transform(df["person_education"])
    df["previous_loan_defaults_on_file"] = le_defaults.transform(df["previous_loan_defaults_on_file"])
    
    # Apply one-hot encoding to the one-hot columns.
    df = pd.get_dummies(df, columns = ["person_gender", "person_home_ownership", "loan_intent"])
    
    # Because the user only inputs one row of data, get_dummies won't work entirely by itself.
    # Therefore, any missing columns necessary for the model (i.e. the remaining one-hot columns not produced by get_dummies here)
    # are imputed as 0, which would be correct.
    expected_dummies = [
        'person_gender_female', 'person_gender_male',
        'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
        'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE'
    ]
    
    # If the column isn't present, add it and make it 0. This is necessary because the Random Forest model expects 23 features.
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = 0
    
    ### An issue originally occurred here, which was that the scaler REQUIRES all columns to be in the same order as they were when
    ### the scaler was fitted. Therefore, the dataframe is rearranged to match how it was in the model's training.
    df = df[["person_age", "person_education", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file", "person_gender_female", "person_gender_male",
    "person_home_ownership_MORTGAGE", "person_home_ownership_OTHER", "person_home_ownership_OWN", "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION", "loan_intent_EDUCATION", "loan_intent_HOMEIMPROVEMENT", "loan_intent_MEDICAL", "loan_intent_PERSONAL",
    "loan_intent_VENTURE"]]
    
    # Now that the DataFrame is correctly ordered in the way the scaler expects, 
    # scale the data using the StandardScaler that was saved when training the model.
    sc.transform(df)
    
    # Pandas has a dedicated "to_numpy" export function, removing the need for a specialised
    # list to be made and converted to a NumPy array.
    arr = df.to_numpy()
    
    # Use the model to make a prediction on the encoded and scaled input.
    prediction = mdl.predict(arr)
    
    # Output the prediction to the console.
    print("Predicted value is %f" % prediction)
    
    # Return the prediction as the result of the POST request.
    return { "Loan Status": int(prediction[0]), "parameters": data_dict }
    
