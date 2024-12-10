import pandas as pd
import redis
import pyarrow as pa

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
	
def write_airline_csv(rdis_conn, serialisation_context):
	# Create a closure to capture the system context
	def inner():
		df = serialisation_context.deserialize(rdis_conn.get("airline_csv"))
		df.to_csv("~/copied_airline_dataframe.csv")
	return inner
