import great_expectations as gx
import pandas as pd

# Create a Data Context
context = gx.get_context()

# Create a Pandas Data Source
datasource = context.sources.add_pandas(name="LoanApproval_DataSource")
data_asset = data_source.add_dataframe_asset
