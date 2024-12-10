

from datetime import datetime, timedelta
from textwrap import dedent

import pandas as pd
import numpy as np
import redis
import pyarrow as pa
from example6.functions import ingest, validate, prepare, train, evaluate



from airflow import DAG
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

# Machine Learning parameters
# We could load these from some external source if we wished
ml_params = {"alpha":2, "l1_ratio":0.5}

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
	"example6",
	default_args=default_args,
	description="A more complex example DAG with more stages",
	schedule_interval=timedelta(days=1),
	start_date=datetime(2021, 10, 10),
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
	
	train_task = PythonOperator(
		task_id="train",
		python_callable=train(redis_conn, serialisation_context, params=ml_params)
	)
	
	evaluate_task = PythonOperator(
	    task_id="evaluate",
	    python_callable=evaluate(redis_conn, serialisation_context)
	)
	ingest_task >> validate_task >> prepare_task >> train_task >> evaluate_task
	
