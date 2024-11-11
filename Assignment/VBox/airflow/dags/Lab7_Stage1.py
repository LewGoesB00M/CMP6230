import redis
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def readIris():
    df = pd.read_csv("/media/sf_CMP6230/airflow/dags/data/iris.csv", names = ["F!", "F2", "F3", "F4", "T1"])
    redis_conn = redis.Redis(host = "127.0.0.1", port = 6379) #, db = redis_db
    table = pa.Table.from_pandas(df)
    output_stream = pa.BufferOutputStream()
    pq.write_table(table, output_stream)
    serialized_data = output_stream.getvalue().to_pybytes()
    redis_conn.set("iris_key", serialized_data)
    # It seems you don't have to manually close a Redis connection?
    
def writeMiniIris():
    redis_conn = redis.Redis(host = "127.0.0.1", port = 6379) #, db = redis_db
    retrieved_data = redis_conn.get("iris_key")
    buffer_reader = pa.BufferReader(retrieved_data)
    retrieved_table = pq.read_table(buffer_reader)
    deserialized_df = retrieved_table.to_pandas()
    # To check it worked, write the first 5 lines to another file.
    deserialized_df.head().to_csv('/media/sf_CMP6230/airflow/dags/data/MiniIris.csv')