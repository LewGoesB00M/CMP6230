from datetime import datetime, timedelta
from textwrap import dedent
import pandas as pd
from sqlalchemy import create_engine
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
# START PythonFunc
# Placeholder space for custom Python functions, in reality you would put them in separate modules
def read_iris_csv():
    # First let's read in our CSV as a single pandas dataframe (it's small enough)
    df_iris = pd.read_csv("iris.csv",
    names=['SepLen','SepWid','PetLen','PetWid','Species'])
    # Then let's put this dataframe into our column store
    eng_conn = create_engine("mysql+pymysql://lewis:dockerVBox@172.17.0.2:3306/Iris")
    df_iris.to_sql("Iris", con=eng_conn, if_exists="replace")
    eng_conn.close()

def write_iris_csv():
    eng_conn = create_engine("mysql+pymysql://lewis:dockerVBox@172.17.0.2:3306/Iris")
    df_iris_tmp = pd.read_sql("Iris", con=eng_conn)
    df_iris_tmp.to_csv('~/duplicateiris.csv')
    eng_conn.close()
# END PythonFunc

default_args = {
    "owner": "Lewis",
    "depends_on_past": False,
    "email": ["lphiggins2004@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,"retries": 0,
    "retry_delay": timedelta(minutes=5)
}
with DAG(
    "example2",
    default_args=default_args,
    description="A simple example DAG",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["example"],
) as dag:
    
    task1 = PythonOperator(
        task_id="get_csv",
        python_callable=read_iris_csv
    )
    task2 = BashOperator(
        task_id="sleep_for_5",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=0,
    )
    
    task3 = PythonOperator(
        task_id="write_csv",
        python_callable=write_iris_csv
    )
    
    task1 >> task2 >> task3