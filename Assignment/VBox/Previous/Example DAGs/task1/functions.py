import sys

import pandas as pd
import redis
import pyarrow as pa
import numpy as np
import pickle as pkl


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


import mlflow
import mlflow.sklearn
import logging


def eval_metrics(actual, pred):
    """
    """
    raise NotImplementedError

def train_test_splitter(data):
    """
    """
    raise NotImplementedError

def get_redwine_table():
    """
    """
    raise NotImplementedError

def ingest(rdis_conn, serialisation_context):
    def inner():
        raise NotImplementedError
    return inner 

def validate(rdis_conn, serialisation_context):
    def inner():
        raise NotImplementedError
    return inner
            
def prepare(rdis_conn, serialisation_context):
    def inner():
        raise NotImplementedError
    return inner
    
def train(rdis_conn, serialisation_context, params = None):
    def inner():
        raise NotImplementedError
    return inner
    
def evaluate(rdis_conn, serialisation_context):
    def inner():
        raise NotImplementedError   
    return inner
