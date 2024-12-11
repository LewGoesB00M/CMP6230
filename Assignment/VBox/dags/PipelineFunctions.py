# When evaluating the model, sys.stdout will be used, which requires sys to be imported.
import sys

# Numpy is needed for its array data types.
import numpy as np

# Pandas is the backbone of this pipeline; it stores the dataset into a "DataFrame",
# and allows for advanced manipulation of the data within.
import pandas as pd

# SQLAlchemy is used to connect to the MariaDB Columnstore database.
from sqlalchemy import create_engine 

# Unlike similar projects, this pipeline does NOT use Apache Arrow.
# This is due to recent changes that make serialisation and Redis storage 
# highly complex.

# A simpler solution was to use DirectRedis, a package that extends the 
# existing functionality of redis-py by serializing and deserializing automatically. 
import redis
from direct_redis import DirectRedis

# MLFlow import
import mlflow
import mlflow.sklearn

# Scikit-learn (sklearn) allows for data encoding and splitting, as well 
# as model training and evaluation.
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Imbalanced-learn (imblearn) allows for the use of SMOTEto oversample
# the imbalanced data.
from imblearn.over_sampling import SMOTE

# Pickle will be used to serialise the ML model.
import pickle as pkl

# URLParse retrieves information about the model from MLFlow.
from urllib.parse import urlparse


# Set the random seed to a specific number to increase reproducibility.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Default variables that are frequently used by the following functions.
DEFAULT_CONN_STRING = "mysql+pymysql://lewis:MLOps6230@172.17.0.2:3306/LoanApproval"
REDIS_CONN_INFO = {"host": "localhost", "port": 6379, "db": 0, "table": "LoanApproval"}
DEFAULT_PATH = "~/CMP6230/loan_data.csv"
DEFAULT_TABLE_NAME = "LoanApproval"

# Creates the SQLAlchemy engine to interface with the MariaDB Columnstore database.
def create_db_context(conn_string = DEFAULT_CONN_STRING):
    return create_engine(conn_string)

# Reads the CSV from its local storage location.
def extract_csv(path = DEFAULT_PATH):
    return pd.read_csv(path)

# Writes data to the Columnstore database using the engine produced by create_db_context.
def write_df_to_db(df, table_name = DEFAULT_TABLE_NAME, if_exists = "replace", conn_string = DEFAULT_CONN_STRING):
    eng_conn = create_db_context(conn_string)
    df.to_sql(table_name, con = eng_conn, if_exists = if_exists, index = False)
    eng_conn.dispose()

# Reads the database from the MariaDB Columnstore database.
def read_df_from_db(conn_string, table_name):
    eng_conn = create_db_context(conn_string)
    return pd.read_sql(table_name, conn_string)

# The main Ingestion function. Reads the CSV into a DataFrame and transforms it 
# before loading it into the Columnstore database via write_df_to_db.
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

# Encodes the data using Label and One-Hot encoders.
def encode(df):
    le = LabelEncoder() # Label encoding converts strings to numbers, and is best suited to ordinal data.
    # In this dataset, person_education is ordinal as a person's level of education goes in order of Bachelors, Masters, etc.
    df["person_education"] = le.fit_transform(df["person_education"])
    
    # One-hot encoding also converts strings to numbers, but to do so it creates new Boolean columns for each
    # of the original values in the column, removing the original column in the process.
    encodedDf = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
    return encodedDf
  
# Splits the dataset into X and Y sets, where the main data is X and the target column is Y.  
def x_y_split(df, target):  
    x = df.drop(target, axis = 1)
    y = df[target]
    return x, y

# Uses SMOTE to balance the dataset.
def oversample(x, y):
    smote = SMOTE()
    x = x.rename(str, axis="columns")  # SMOTE doesn't work with column names unless all columns are specifically str type.
    return smote.fit_resample(x, y)

# Splits the dataset into training and testing sets for supervised learning.
def train_test_splitter(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = RANDOM_STATE)
    return train_x, test_x, train_y, test_y

# The main Preprocessing function.
# Retrieves the DataFrame from the Columnstore, removes outliers,
# then encodes, balances, splits and stores the data in the Redis memory store.
def preprocess(from_conn, to_redis = REDIS_CONN_INFO):
    def inner():
        eng_conn = create_db_context(from_conn)
        df = read_df_from_db(from_conn, REDIS_CONN_INFO["table"])

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
        r.set("trainX", train_x)
        r.set("testX", test_x)
        r.set("trainY", train_y)
        r.set("testY", test_y)
    return inner
    

# The main Training function.
# Reads the DataFrame from Redis, scales the data, then 
# trains a Random Forest Classifier model on the data. 
def train(params, from_redis = REDIS_CONN_INFO):
    def inner():
        r = DirectRedis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])
        
        # Deserialize the dataframes from Redis.
        train_x = r.get("trainX")
        test_x = r.get("testX")
        train_y = r.get("trainY")
        test_y = r.get("testY")
        
        # Scale the data.
        sc = StandardScaler()
        train_x = sc.fit_transform(train_x)
        test_x = sc.transform(test_x)
        
        # Start an MLFlow run.
        run = mlflow.start_run()
        
        # Instantiate and train the model based on the given parameters.
        mdl = RandomForestClassifier(n_estimators = params["n_estimators"])
        mdl.fit(train_x, train_y)

        # Pickle is used to seralize the model.
        serialized_mdl = pkl.dumps(mdl)
        r.set("LoanApproval_trained_mdl", serialized_mdl)

	# Direct-Redis serializes the parameters and run ID, to be used in Evaluation.
        r.set("LoanApproval_trained_params", params)
        r.set("LoanApproval_trained_run_id", run.info.run_id)
        
        # Pause the run. It will be resumed during evaluation.
        mlflow.end_run()
    return inner
    
# Gets the evaluation metrics of the model by comparing its predictions to the real data.
# Specifically gets the accuracy, precision and F1-Scores.
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    
    return accuracy, precision, f1
    
# The main Evaluation function.
# Gets the model, parameters and testing set from Redis,
# and then makes predictions and evaluates the results, saving them to MLFlow.
def evaluate(from_redis = REDIS_CONN_INFO):
    def inner():
        r = DirectRedis(host = REDIS_CONN_INFO["host"], port = REDIS_CONN_INFO["port"], db = REDIS_CONN_INFO["db"])

	# Retrieve the serialised model from Redis.
        mdl = pkl.loads(r.get("LoanApproval_trained_mdl"))
            
        # Getting the parameters
        params = r.get("LoanApproval_trained_params")
        n_estimators = params["n_estimators"]

        # Get the testing set from Redis.
        test_x = r.get("testX")
        test_y = r.get("testY")
        
        # Restart the MLFlow run.
        run_id = r.get("LoanApproval_trained_run_id")
        mlflow.start_run(run_id = run_id)
        
        # Predict with the model.
        predicted_qualities = mdl.predict(test_x)

        # Evaluate the results.     
        accuracy, precision, f1 = eval_metrics(test_y, predicted_qualities)

        # Output the metrics to stdout, which is somewhat like 
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

        # Logging the model
        # Model registering with a name doesn't work with a file-type filestore, so we must ensure it isn't one.
        if tracking_url_type_filestore != "file":            
            # Registering the model
            mlflow.sklearn.log_model(mdl, "model", registered_model_name="LoanApproval_RandomForestModel")
        else:
            mlflow.sklearn.log_model(mdl, "model")

        # End the current experiment run
        mlflow.end_run()
    return inner    
    
    

# DEFAULT_COLUMNS = ["person_age","person_gender","person_education","person_income","person_emp_exp","person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file","loan_status"]
