# My understanding of DAG files is that you wouldn't declare all the functions here,
# but rather in other files to be imported as modules. In doing so, you'll massively
# reduce clutter.
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from Lab7_Stage1 import readIris, writeMiniIris

default_args = {
    "owner": "Lewis",
    "depends_on_past": False,
    "email": ["lphiggins2004@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5)
}


with DAG(
    "Lab7",
    default_args=default_args,
    description="Lab 7 - Experimenting with Redis and Arrow.",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["Lab"],
) as dag:
    
    task1 = PythonOperator(
        task_id="get_csv",
        python_callable=readIris
    )
    task2 = BashOperator(
        task_id="sleep_for_3",
        depends_on_past=False,
        bash_command="sleep 3",
        retries=0,
    )
    
    task3 = PythonOperator(
        task_id="write_csv",
        python_callable=writeMiniIris
    )
    
    task1 >> task2 >> task3