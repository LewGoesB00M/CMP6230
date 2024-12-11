import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
import sklearn as skl
import pyarrow as pa
import pyarrow.parquet as pq
import redis
from direct_redis import DirectRedis

import great_expectations as gx

# MLFlow import
import mlflow
import mlflow.sklearn

# Set up logging of important data
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE

import pickle as pkl

from urllib.parse import urlparse


GX_DIR = "/home/lewis/PipelineGX/gx"
RANDOM_STATE = 42
DEFAULT_CONN_STRING = "mysql+pymysql://lewis:MLOps6230@172.17.0.2:3306/LoanApproval"
REDIS_CONN_INFO = {"host": "localhost", "port": 6379, "db": 0, "table": "LoanApproval"}
DEFAULT_PATH = "~/CMP6230/loan_data.csv"
# DEFAULT_COLUMNS = ["person_age","person_gender","person_education","person_income","person_emp_exp","person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file","loan_status"]
DEFAULT_TABLE_NAME = "LoanApproval"

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
    def inner():
            
        df_extracted = extract_csv(from_path)
        
        # Transformation
        df_extracted["person_income"] = df_extracted["person_income"].astype(int)
        df_extracted["person_age"] = df_extracted["person_age"].astype(int)
        df_extracted["loan_amnt"] = df_extracted["loan_amnt"].astype(int)
        df_extracted["cb_person_cred_hist_length"] = df_extracted["cb_person_cred_hist_length"].astype(int)
        
        write_df_to_db(df_extracted)
    return inner

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
    return train_x, test_x, train_y, test_y

def preprocess(from_conn, to_redis = REDIS_CONN_INFO):
    def inner():
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
        train_x, test_x, train_y, test_y = train_test_splitter(x, y)

        # Serialize the dataframes and send them to Redis to be reimported in model development
        r = DirectRedis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
        
    #     train_buffer = pa.serialize_pandas(train_x)
        r.set("trainX", train_x)
        
    #     test_buffer = pa.serialize_pandas(test_x)
        r.set("testX", test_x)

    #     y_table = pa.Table.from_pandas(train_y)
    #     output_stream = pa.BufferOutputStream()
    #     pq.write_table(y_table, output_stream)
    #     serialized_y = output_stream.getvalue().to_pybytes()
    #     serialized_y = train_y.to_pickle()
        r.set("trainY", train_y)
        
    #     y_table = pa.Table.from_pandas(test_y)
    #     output_stream = pa.BufferOutputStream()
    #     pq.write_table(y_table, output_stream)
    #     serialized_y = output_stream.getvalue().to_pybytes()
        r.set("testY", test_y)
    return inner
  
def train(params, from_redis = REDIS_CONN_INFO):
    def inner():
        r = DirectRedis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
        
        # Deserialize the dataframes from Redis.
        train_x = r.get("trainX")
    #     train_x = pa.deserialize_pandas(train_x_buffer)
        
        test_x = r.get("testX")
    #     test_x = pa.deserialize_pandas(test_x_buffer)
        
        train_y = r.get("trainY")
    #     train_y = pa.deserialize_pandas(train_y_buffer)
        
        test_y = r.get("testY")
    #     test_y = pa.deserialize_pandas(test_y_buffer)
        
        # Scale the data.
        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)
        
        # Start MLFlow.
        run = mlflow.start_run()
        
        mdl = RandomForestClassifier(n_estimators = params["n_estimators"])
        mdl.fit(train_x, train_y)

        # Pickle is used to seralize the model.
        serialized_mdl = pkl.dumps(mdl)
        r.set("LoanApproval_trained_mdl", serialized_mdl)

        r.set("LoanApproval_trained_params", params)
        
        r.set("LoanApproval_trained_run_id", run.info.run_id)
    return inner
    
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    
    return accuracy, precision, f1
    
def evaluate(from_redis = REDIS_CONN_INFO):
    def inner():
        r = DirectRedis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
        
        tmp = r.get("LoanApprovalModel")
        print(tmp)
        
        mdl = pkl.loads(r.get("LoanApproval_trained_mdl"))
            
        # Getting the parameters
        params = r.get("LoanApproval_trained_params")
        n_estimators = params["n_estimators"]
        # Another parameter might be good.

        # Getting the run id for MLFlow to continue the experiment
        run_id = r.get("LoanApproval_trained_run_id")

        # Resume the previously started experiment run
        # run = mlflow.start_run(run_id=run_id)

        # Perform the prediction
        test_x = r.get("testX")
        test_y = r.get("testY")
        predicted_qualities = mdl.predict(test_x)

        # Evaluate the results     
        accuracy, precision, f1 = eval_metrics(test_y, predicted_qualities)

        # Output the metrics to the stdout   
        # (this is somewhat equivalent to print)
        sys.stdout.write("Random Forest model (n_estimators: %f):\n" % (n_estimators))
        sys.stdout.write("  Accuracy: %s\n" % accuracy)
        sys.stdout.write("  Precision: %s\n" % precision)
        sys.stdout.write("  F1: %s\n" % f1)

        # Log the parameters to mlflow
        mlflow.log_param("n_estimators", n_estimators)

        # Log the metrics to mlflow
        mlflow.log_metric("accuracy", accuracy) 
        mlflow.log_metric("precision", precision) 
        mlflow.log_metric("f1", f1)     
        tracking_url_type_filestore = urlparse(mlflow.get_tracking_uri()).scheme

        # model logging
        # Model registry does not work with file store
        if tracking_url_type_filestore != "file":            
            # Registering the model
            # Please refer to the documentation for further information:
            # https://mlflow.org/docs/la/model-registry.html#api-workflow
            mlflow.sklearn.log_model(mdl, "model", registered_model_name="LoanApproval_RandomForestModel")
        else:
            mlflow.sklearn.log_model(mdl, "model")

        # End the current experiment run
        mlflow.end_run()
    return inner    