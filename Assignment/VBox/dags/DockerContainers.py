from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
	"owner": "Lewis Higgins",
    "description": "A DAG to start the necessary Docker containers - MariaDB Columnstore on port 3306 and Redis on port 6379.",
	"depends_on_past": False,
	"email": ["Lewis.Higgins@mail.bcu.ac.uk"],
	"email_on_failure": False,
	"email_on_retry": False,
	"retries": 1,
	"retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 12, 10) # Year, Month, Day format.
}

with DAG("DockerContainers", default_args=default_args, schedule_interval=timedelta(days=1), catchup=False, tags=["Pipeline"]) as dag:
	Columnstore = BashOperator(
		task_id = "start_columnstore",
		bash_command = "docker start Columnstore"
	)
	
	Redis = BashOperator(
		task_id="start_redis",
		bash_command = "docker start Redis"
	)
	
	Columnstore >> Redis