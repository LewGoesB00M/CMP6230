from datetime import datetime, timedelta
import logging

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 
import sklearn as skl
import pyarrow as pa
import pyarrow.parquet as pq
import redis
from direct_redis import DirectRedis
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator
from great_expectations.core.batch import BatchRequest
from great_expectations.data_context.types.base import DataContextConfig, CheckpointConfig

# MLFlow import
import mlflow
import mlflow.sklearn


from PipelineFunctions import * 

# Set up logging of important data
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://localhost:5000")


default_args = {
	"owner": "Lewis Higgins",
    "description": "The complete DAG for the MLOps Pipeline implementation.",
	"depends_on_past": False,
	"email": ["Lewis.Higgins@mail.bcu.ac.uk"],
	"email_on_failure": False,
	"email_on_retry": False,
	"retries": 1,
	"retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 12, 11) # Year, Month, Day format.
}

GX_DIR = "/home/lewis/PipelineGX/gx"
RANDOM_STATE = 42
DEFAULT_CONN_STRING = "mysql+pymysql://lewis:MLOps6230@172.17.0.2:3306/LoanApproval"
REDIS_CONN_INFO = {"host": "localhost", "port": 6379, "db": 0, "table": "LoanApproval"}
DEFAULT_PATH = "~/CMP6230/loan_data.csv"
# DEFAULT_COLUMNS = ["person_age","person_gender","person_education","person_income","person_emp_exp","person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file","loan_status"]
DEFAULT_TABLE_NAME = "LoanApproval"

with DAG(
    "MLOpsPipeline", 
    description = "The complete DAG for the MLOps pipeline implementation.",
    default_args=default_args, 
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["Pipeline"]
) as dag:    
	IngestTask = PythonOperator(
    	task_id = "etl", 
     	python_callable = extract_transform_load()
    )

	ValidateTask = GreatExpectationsOperator(
		task_id = "GX_Validation",
		data_context_root_dir = GX_DIR,
		checkpoint_name= "LoanApproval_Original_checkpoint",
		return_json_dict=True
	)

	PreprocessingTask = PythonOperator(
     		task_id = "preprocessing", 
     		python_callable = preprocess(DEFAULT_CONN_STRING)
    )

	TrainingTask = PythonOperator(
		task_id = "training", 
		python_callable = train(params = {"n_estimators": 200}))
 
	EvaluationTask = PythonOperator(task_id = "evaluation", python_callable = evaluate())
        

IngestTask >> ValidateTask >> PreprocessingTask >> TrainingTask >> EvaluationTask


