import sys

import pandas as pd
import redis
import pyarrow as pa
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


import mlflow
import mlflow.sklearn
import logging


def eval_metrics(actual, pred):
    """
    Evaluates the success of a model by comparing the predicted values with the actual target values
    
    Shorthand:
        pd.DF = pandas.core.frame.DataFrame  
    
    Parameters:
        actual (pd.DF) : A pandas dataframe containing the target values (Y) 
    
        pred (pd.DF) : A pandas dataframe containing the predicted values (Y)
    
    Returns:
        (float * float * float) : 
            A triple containing (rmse, mae, r2)
            rmse being the root mean squared error, mae being the mean absolute error and r2 being the r squared score
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_test_splitter(data):
    """
    Splits our data into two triples, train and test
    With each triple having the format: (all, x_features, y_features)
    
    Shorthand:
        pandas.core.frame.DataFrame = pd.DF
    
    Parameters:
        data (pd.DF): A pandas dataframe containing the retrieved sample data to split
    
    Returns:
        ((pd.DF * pd.DF * pd.DF) * (pd.DF * pd.DF * pd.DF)) : 
            A tuple of truples containing ((train, train_x, train_y), (test, test_x, test_y)
            with train being the complete training sample (including x and y values) 
            and test being the complete testing sample (including x and y values)
    """
	# Split the data into training and test sets. 
	# What ratio does this function use to split into training and tests by default?
    train, test = train_test_split(data)    
    # Our predicted feature is "quality" what kind of data is it?
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return ((train, train_x, train_y), (test, test_x, test_y))
    
def get_csv():
    """
    Retrieves a CSV file based on the URL we have hardcoded
    
    Shorthand:
        pandas.core.frame.DataFrame = pd.DF
    
    Parameters:
        None
        
    Returns:
        pd.DF : 
            A dataframe containing the results of reading the CSV from the specified URL.
            In this case a data set detailing properties of red wine (including a rating of their quality)
    """
    csv_url =\
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(csv_url, sep=';')
        return df
    except Exception as e:
        logger.exception(
            "Error: Unable to download the red wine dataset, please check your network connectivity. %s", e) 
            
def ingest(rdis_conn, serialisation_context):
    def inner():
        df = get_csv()
        
        # Serialise the data to redis, in practice during data ingestion we would likely store
        # in another DB system such as MariaDB columnstore
        serialized_data = serialisation_context.serialize(df).to_buffer().to_pybytes()
        rdis_conn.set("wine_ingested", serialized_data)
    return inner 
        
        
def validate(rdis_conn, serialisation_context):
    def inner():
        # Get data from redis
        df = serialisation_context.deserialize(rdis_conn.get("wine_ingested"))
        
        # Do other validation processing
        
        # Store data in redis
        serialized_data = serialisation_context.serialize(df).to_buffer().to_pybytes()
        rdis_conn.set("wine_validated", serialized_data)    
    return inner
            
def prepare(rdis_conn, serialisation_context):
    # We create a closure function to allow us to store the external state needed when calling the function
    def inner():
        # Get data from redis
        df = serialisation_context.deserialize(rdis_conn.get("wine_validated"))
        
        # Do the actual data preparation stuff here
       
        # Split the data
        tpl = train_test_splitter(df)
        
        # Serialise the data to redis
        serialized_data = serialisation_context.serialize(tpl).to_buffer().to_pybytes()
        rdis_conn.set("wine_prepared", serialized_data)
    return inner
    
def train_and_evaluate(rdis_conn, serialisation_context, alpha=0.5, l1_ratio=0.5):
    def inner():
        # Get data from redis
        data = serialisation_context.deserialize(rdis_conn.get("wine_prepared"))

        (training, testing) = data
        ((_, train_x, train_y), (_, test_x, test_y)) = (training, testing)
        
        # Initiating mlflow experiment run
        with mlflow.start_run():
        	# Create model builder
            mdl = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            
            # Fit the regression model
            mdl.fit(train_x, train_y)        
            predicted_qualities = mdl.predict(test_x)        
            rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)     
            
            # Output the metrics to the stdout   
            # (this is somewhat equivalent to print)
            sys.stdout.write("Elasticnet model (alpha=%f, l1_ratio=%f):\n" % (alpha, l1_ratio))
            sys.stdout.write("  RMSE: %s\n" % rmse)
            sys.stdout.write("  MAE: %s\n" % mae)
            sys.stdout.write("  R2: %s\n" % r2)
            
            # Log the parameters to mlflow
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            
            # Log the metrics to mlflow
            mlflow.log_metric("rmse", rmse) # root mean square error
            mlflow.log_metric("r2", r2) # r squared
            mlflow.log_metric("mae", mae) # mean absolute error       
            tracking_url_type_filestore = urlparse(mlflow.get_tracking_uri()).scheme
            
            # Model logging
            # Model registry doesn't work with the file store system
            if tracking_url_type_filestore != "file":            
                # Registering the model
                # Please refer to the documentation for further information:
                # https://mlflow.org/docs/la/model-registry.html#api-workflow
                mlflow.sklearn.log_model(mdl, "model", registered_model_name="Winequality_ElasticnetModel")
            else:
                mlflow.sklearn.log_model(mdl, "model")        
    return inner
    

