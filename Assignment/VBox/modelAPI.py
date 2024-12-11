from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Loading our model from a file
# The model's pickled file might be at a different location on your filesystem
path = "./mlartifacts/0/0a814dd2a1e340a28890f70b17d98b46/artifacts/model/model.pkl"
pickle_input = open(path, "rb")
mdl = pickle.load(pickle_input)

columns = "person_age, person_education, person_income, person_emp_exp, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, loan_status, person_gender_female, person_gender_male, person_home_ownership_MORTGAGE, person_home_ownership_OTHER, person_home_ownership_OWN, person_home_ownership_RENT, loan_intent_DEBTCONSOLIDATION, loan_intent_EDUCATION, loan_intent_HOMEIMPROVEMENT, loan_intent_MEDICAL, loan_intent_PERSONAL, loan_intent_VENTURE, previous_loan_defaults_on_file_No, previous_loan_defaults_on_file_Yes"
    
class LoanInput(BaseModel):
    person_age: int
    person_education: int
    person_income: int
    person_emp_exp: int 
    loan_amnt: int
    loan_int_rate: float 
    loan_percent_income: float 
    cb_person_cred_hist_length: int
    credit_score: int
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
    previous_loan_defaults_on_file_No: int
    previous_loan_defaults_on_file_Yes: int
    
    # A highly unfortunate side effect of one-hot encoding.

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
    previous_loan_defaults_on_file_No: int
    previous_loan_defaults_on_file_Yes: int
   
# Leaving this in from the example
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# Our model serving endpoint    
@app.post("/loan/predict/single")
def predict_loan_single(data: LoanInput):
    data_dict = data.dict()

    age = data_dict["person_age"]
    education = data_dict["person_education"]
    income = data_dict["person_income"]
    empexp = data_dict["person_emp_exp"]
    loanamnt = data_dict["loan_amnt"]
    interest = data_dict["loan_int_rate"]
    pctincome = data_dict["loan_percent_income"]
    credhist = data_dict["cb_person_cred_hist_length"]
    credscore = data_dict["credit_score"] 
    female = data_dict["person_gender_female"]
    male = data_dict["person_gender_male"]
    mortgage = data_dict["person_home_ownership_MORTGAGE"]
    other  = data_dict["person_home_ownership_OTHER"]
    own = data_dict["person_home_ownership_OWN"]
    rent = data_dict["person_home_ownership_RENT"]
    debtcons = data_dict["loan_intent_DEBTCONSOLIDATION"]
    loanedu = data_dict["loan_intent_EDUCATION"]
    homeimp = data_dict["loan_intent_HOMEIMPROVEMENT"]
    medic = data_dict["loan_intent_MEDICAL"]
    personal = data_dict["loan_intent_PERSONAL"]
    venture = data_dict["loan_intent_VENTURE"]
    defaulted = data_dict["previous_loan_defaults_on_file_No"] # Re-encode this one to be ordinal. LabelEncoder.
    notDefaulted = data_dict["previous_loan_defaults_on_file_Yes"]
    lst = [age, education, income, empexp, loanamnt, interest, pctincome, credhist, credscore, female, male, mortgage, other, own, rent, debtcons, loanedu, homeimp, medic, personal, venture, defaulted, notDefaulted]
    arr = np.asarray(lst).reshape(1, -1)
    prediction = mdl.predict(arr)
    print("Predicted value is %f" % prediction)
    return { "Loan Status": int(prediction[0]), "parameters": data_dict }
    
