

from datetime import datetime, timedelta
from textwrap import dedent

import pandas as pd
import redis
import pyarrow as pa
from example4.test import read_airline_csv, write_airline_csv


from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator



redis_conn = redis.Redis(host="127.0.0.1", port=6379) # , db=redis_db
serialisation_context = pa.default_serialization_context()

#	START PythonFunc
# 	Placeholder space for custom Python functions, in reality you would put them in separate modules



#	END PythonFunc 


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
	"example4",
	default_args=default_args,
	description="A simple example DAG",
	schedule_interval=timedelta(days=1),
	start_date=datetime(2021, 10, 10),
	catchup=False,
	tags=["example"],
) as dag:
	task1 = PythonOperator(
		task_id="get_csv",
		python_callable=read_airline_csv(redis_conn, serialisation_context)
	)
	
	task2 = BashOperator(
		task_id="sleep_for_1",
		depends_on_past=False,
		bash_command="sleep 1",
		retries=0,
	)
	task3 = PythonOperator(
		task_id="write_csv",
		python_callable=write_airline_csv(redis_conn, serialisation_context)
	)
	
	task1 >> task2 >> task3
	
