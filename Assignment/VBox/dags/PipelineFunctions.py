import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Importing mlflow and its sklearn component
import mlflow
import mlflow.sklearn

def create_db_context(conn_string = DEFAULT_CONN_STRING):
    return create_engine(conn_string)

def extract_csv(path = DEFAULT_PATH):
    return pd.read_csv(path)

def write_df_to_db(df, table_name = DEFAULT_TABLE_NAME, if_exists = "replace", conn_string = DEFAULT_CONN_STRING):
    eng_conn = create_db_context(conn_string)
    df.to_sql(table_name, con = eng_conn, if_exists = if_exists, index = False)
    eng_conn.dispose()
    
def read_df_from_db(conn_string, table_name):
    eng_conn = create_db_context(conn_string)
    return pd.read_sql(table_name, conn_string)

def extract_transform_load(from_path = DEFAULT_PATH, to_conn = DEFAULT_CONN_STRING):
    df_extracted = extract_csv(from_path)
    
    # Transformation
    df_extracted["person_income"] = df_extracted["person_income"].astype(int)
    df_extracted["person_age"] = df_extracted["person_age"].astype(int)
    df_extracted["loan_amnt"] = df_extracted["loan_amnt"].astype(int)
    df_extracted["cb_person_cred_hist_length"] = df_extracted["cb_person_cred_hist_length"].astype(int)
    
    write_df_to_db(df_extracted)

def encode(df):
    # Encode the data using Label and One-Hot encoders.
    le = LabelEncoder() # Label encoding converts strings to numbers, and is best suited to ordinal data.
    # In this dataset, person_education is ordinal as a person's level of education goes in order of Bachelors, Masters, etc.
    df["person_education"] = le.fit_transform(df["person_education"])
    
    # One-hot encoding also converts strings to numbers, but to do so it creates new Boolean columns for each
    # of the original values in the column, removing the original column in the process.
    encodedDf = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
#     print(encodedDf.columns)
#     print(encodedDf.shape)
    return encodedDf
    
def x_y_split(df, target):
    # Split the dataset into X and Y sets, where the main data is X and the target column is Y.
    x = df.drop(target, axis = 1)
    y = df[target]
    return x, y

def oversample(x, y):
    smote = SMOTE()
    x = x.rename(str, axis="columns")  # SMOTE doesn't work unless this line is added.
    return smote.fit_resample(x, y)

def train_test_splitter(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = RANDOM_STATE)
    trainingSet = pd.concat([train_x, train_y], axis = 1)
    print(trainingSet.shape)
    testingSet = pd.concat([test_x, test_y], axis = 1)
    print(testingSet.shape)
    return trainingSet, testingSet

def preprocessing(from_conn, to_redis = REDIS_CONN_INFO):
    eng_conn = create_db_context(from_conn)
    df = read_df_from_db(from_conn, REDIS_CONN_INFO["table"])
    
    # Preprocessing
    # person_income varies wildly, so let's drop some of them.
    outlierHighIncome  = df["person_income"].quantile(0.98)
    df = df[(df["person_income"] < outlierHighIncome)]
    
    # Also remove rows with unlikely ages.
    df = df[(df["person_age"] < 85)]
    
    # Encode categorical rows.
    df = encode(df)
    
    # Split into X and Y sets.
    x, y = x_y_split(df, "loan_status")
    
    # Apply SMOTE to balance the dataset.
    x, y = oversample(x, y)
    
    # Split the data into training and testing sets
    train, test = train_test_splitter(x, y)

    # Serialize the dataframes and send them to Redis to be reimported in model development
    r = redis.Redis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
    
    train_buffer = pa.serialize_pandas(train)
    r.set(REDIS_CONN_INFO["train"], train_buffer.to_pybytes())
    
    test_buffer = pa.serialize_pandas(test)
    r.set(REDIS_CONN_INFO["test"], test_buffer.to_pybytes())
    
def training(from_redis = REDIS_CONN_INFO):
    r = redis.Redis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
    
    # Deserialize the dataframes from Redis.
    train_buffer = r.get(REDIS_CONN_INFO["train"])
    train = pa.deserialize_pandas(train_buffer)
    
    test_buffer = r.get(REDIS_CONN_INFO["test"])
    test = pa.deserialize_pandas(test_buffer)
    
    # Training
    # Then store it to MLFlow (somehow...)