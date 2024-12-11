from datetime import datetime, timedelta
import logging
import numpy as np

from airflow.decorators import dag, task, task_group
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator
from great_expectations.core.batch import BatchRequest
from great_expectations.data_context.types.base import DataContextConfig, CheckpointConfig

# MLFlow import
import mlflow


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

GX_DIR = "/home/lewis/PipelineGX/gx" # I think this is the directory? 
RANDOM_STATE = 42
DEFAULT_CONN_STRING = "mysql+pymysql://lewis:MLOps6230@172.17.0.3:3306/LoanApproval"
REDIS_CONN_INFO = {"host": "localhost", "port": 6379, "db": 0, "table": "LoanApproval"}
DEFAULT_PATH = "~/CMP6230/loan_data.csv"
# DEFAULT_COLUMNS = ["person_age","person_gender","person_education","person_income","person_emp_exp","person_home_ownership","loan_amnt","loan_intent","loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file","loan_status"]
DEFAULT_TABLE_NAME = "LoanApproval"

@dag(dag_id = "MLOpsPipeline", default_args=default_args, schedule_interval=timedelta(days=1), catchup=False, tags=["Pipeline"])

@task_group(group_id = "DockerSetup")
def Docker():
	Columnstore = BashOperator(
		task_id = "start_columnstore",
		bash_command = "docker start Columnstore"
	)
	
	Redis = BashOperator(
		task_id="start_redis",
		bash_command = "docker start Redis"
	)
 
	Columnstore >> Redis

@task_group(group_id = "Ingestion")
def Ingestion():
    validate = GreatExpectationsOperator(
			task_id = "GX_Validation",
			data_context_root_dir = GX_DIR,
			checkpoint_name= "LoanApproval_Original_checkpoint",
			return_json_dict=True
		)
    
    etl = PythonOperator(task_id = "create_db_context", python_callable = extract_transform_load)
    
    validate >> etl

@task_group(group_id = "Preprocessing")
def Preprocessing():
    preprocess = PythonOperator(task_id = "preprocessing", python_callable = preprocessing(DEFAULT_CONN_STRING))
    
    preprocess

def ModelDevelopment():

Docker() >> Ingestion() >> Preprocessing() >> ModelDevelopment() >> ModelDeployment() >> ModelEvaluation() 


