import airflow
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

from datetime import datetime
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from PIL import Image
import pandas as pd
import random
from tqdm import tqdm
import timm

import sys
sys.path.append('../')
from ml_pipeline import train_ml_model
from ml_pipeline import evaluate_ml_model
import pickle

#initiate parameter
default_args = {
    'owner': 'default_user',
    'start_date': airflow.utils.dates.days_ago(1),
    'depends_on_past': False,

    'email': ['your-email@gmail.com'],
    'email_on_failure': True,

    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=30),
}

#create DAGs
dag = DAG(
        'ml_workflow',
        default_args=default_args,
        schedule_interval=timedelta(days=1),
        )

#define funtion
def TRAIN_MODEL_WRAPPER():
    print('----- TRAIN START -----')
    train_ml_model()
    print('----- TRAIN STOP -----')

def EVALUATE_MODEL_WRAPPER():
    print('----- EVALUATE START -----')
    evaluate_ml_model()
    print('----- EVALUATE STOP -----')

#create task
TRAIN_MODEL_TASK	= PythonOperator(
                    task_id='TRAIN_MODEL_TASK', 
                    python_callable=TRAIN_MODEL_WRAPPER,
                    dag=dag)

EVALUATE_MODEL_TASK	= PythonOperator(
                    task_id='EVALUATE_MODEL_TASK', 
                    python_callable=EVALUATE_MODEL_WRAPPER,
                    dag=dag)

#set dependencies
TRAIN_MODEL_TASK >> EVALUATE_MODEL_TASK