

from datetime import datetime, timedelta
from textwrap import dedent

import pandas as pd
import numpy as np
import redis
import pyarrow as pa
from example5.functions import ingest, validate, prepare, train_and_evaluate



import airflow
from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

# Importing mlflow and its sklearn component
import mlflow
import mlflow.sklearn
import logging

# Set up logging of important data
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

# Setting up the ability to serialise data into redis between tasks
redis_conn = redis.Redis(host="127.0.0.1", port=6379) 
serialisation_context = pa.default_serialization_context()


# For greater predictability with randomised operations
np.random.seed(40) 

default_args = {
	"owner": "<your name>",
	"depends_on_past": False,
	"email": ["studentemail@mail.bcu.ac.uk"],
	"email_on_failure": False,
	"email_on_retry": False,
	"retries": 0,
	"retry_delay": timedelta(minutes=5)
}

with DAG(
	"example5",
	default_args=default_args,
	description="A more complex example DAG with more stages",
	schedule_interval=timedelta(days=1),
	start_date=airflow.utils.dates.days_ago(1),
	catchup=False,
	tags=["example"],
) as dag:
	ingest_task = PythonOperator(
		task_id="ingest",
		python_callable=ingest(redis_conn, serialisation_context)
	)
	
	validate_task = PythonOperator(
		task_id="validate",
		python_callable=validate(redis_conn, serialisation_context),
	)
	prepare_task = PythonOperator(
		task_id="prepare",
		python_callable=prepare(redis_conn, serialisation_context)
	)
	
	train_and_evaluate_task = PythonOperator(
		task_id="train_and_evaluate",
		python_callable=train_and_evaluate(redis_conn, serialisation_context)
	)
	
	
	ingest_task >> validate_task >> prepare_task >> train_and_evaluate_task
	
