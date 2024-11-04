

from datetime import datetime, timedelta
from textwrap import dedent

import pandas as pd
import dask as dsk
import dask.dataframe as dd
import dask.distributed as ddst
import redis
import pyarrow as pa


from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

from example3 import functions

#	START PythonFunc
# 	Placeholder space for custom Python functions, in reality you would put them in separate modules

redis_conn = redis.Redis(host="127.0.0.1", port=6379) # , db=redis_db
context = pa.default_serialization_context()

def read_airline_csv(rdis_conn, serialisation_context):
	# Create a closure to capture the system context
	def inner():
		# First let's read in our CSV as a single pandas dataframe (it's small enough)
		df = pd.read_csv("~/lax_to_jfk/lax_to_jfk.csv")
		
		# Then let's put this dataframe into the redis key value store
		# This involves a process of serialization (allowing our in-memory objects to be stored outside of Python)
		# context.serialize(df).to_buffer().to_pybytes(ddist.serialize(df))
		serialized_data = serialisation_context.serialize(df).to_buffer().to_pybytes()
		rdis_conn.set("airline_csv", serialized_data)
	return inner

def write_airline_csv(serialisation_context):
	# Create a closure to capture the system context
	def inner():
		df = serialisation_context.deserialize(redis_conn.get("airline_csv"))
		df.to_csv("~/copied_airline_dataframe.csv")
	return inner
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
	"example3",
	default_args=default_args,
	description="A simple example DAG",
	schedule_interval=timedelta(days=1),
	start_date=datetime(2021, 10, 10),
	catchup=False,
	tags=["example"],
) as dag:
	task1 = PythonOperator(
		task_id="get_csv",
		python_callable=read_airline_csv(redis_conn, context)
	)
	
	task2 = BashOperator(
		task_id="sleep_for_1",
		depends_on_past=False,
		bash_command="sleep 1",
		retries=0,
	)
	task3 = PythonOperator(
		task_id="write_csv",
		python_callable=write_airline_csv(context)
	)
	
	task1 >> task2 >> task3
	
