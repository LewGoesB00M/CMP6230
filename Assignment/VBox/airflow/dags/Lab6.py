from datetime import datetime, timedelta
from textwrap import dedent
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

# START PythonFunc
# Placeholder space for custom Python functions, in reality you would put them in separate modules
def read_heart_csv():
    df_iris = pd.read_csv("/media/sf_CMP6230/airflow/dags/data/Heart.csv")
    
    # Then let's put this dataframe into our column store
    eng_conn = create_engine("mysql+pymysql://lewis:dockerVBox@172.17.0.2:3306/Heart")
    
    # THIS LINE IS MANDATORY DUE TO THIS USING SQLALCHEMY 1.4, WHICH USES "LAZY" CONNECTIONS.
    # THIS MEANS THAT WITHOUT THIS, IT WILL ALWAYS SAY "Engine object has no attribute X" BECAUSE IT HASN'T CONNECTED YET.
    eng_conn = eng_conn.connect()
    df_iris.to_sql("Heart", con=eng_conn, if_exists="replace")
    eng_conn.close()

def write_heart_csv():
    eng_conn = create_engine("mysql+pymysql://lewis:dockerVBox@172.17.0.2:3306/Heart")
    
    # THIS LINE IS MANDATORY DUE TO THIS USING SQLALCHEMY 1.4, WHICH USES "LAZY" CONNECTIONS.
    # THIS MEANS THAT WITHOUT THIS, IT WILL ALWAYS SAY "Engine object has no attribute X" BECAUSE IT HASN'T CONNECTED YET.
    eng_conn = eng_conn.connect()
    
    df_iris_tmp = pd.read_sql("Heart", con=eng_conn)
    df_iris_tmp.to_csv('/media/sf_CMP6230/airflow/dags/data/HeartDuplicate.csv')
    eng_conn.close()
# END PythonFunc

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
    "Lab6",
    default_args=default_args,
    description="Lab 6 - Exercise 3, importing and saving another dataset via an Airflow DAG.",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["Lab"],
) as dag:
    
    task1 = PythonOperator(
        task_id="get_csv",
        python_callable=read_heart_csv
    )
    task2 = BashOperator(
        task_id="sleep_for_3",
        depends_on_past=False,
        bash_command="sleep 3",
        retries=0,
    )
    
    task3 = PythonOperator(
        task_id="write_csv",
        python_callable=write_heart_csv
    )
    
    task1 >> task2 >> task3