from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from kaggle.api.kaggle_api_extended import KaggleApi
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
import os
import zipfile

def download_kaggle_data():
    os.environ['KAGGLE_USERNAME'] = 'mykytavarenykucu'
    os.environ['KAGGLE_KEY'] = '1c07c1a62be9127fb1542de84ba31f9a'

    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    competition = 'tweet-sentiment-extraction'
    api.competition_download_files(competition, path='/tmp')

    # Extract the zip file
    with zipfile.ZipFile(f'/tmp/{competition}.zip', 'r') as zip_ref:
        zip_ref.extractall('/tmp')

def upload_to_minio():
    # Initialize MinIO client
    client = Minio(
        'minio:9000',
        access_key='2i2vYeYLmQ3F1mRMg7ND',
        secret_key='OBll6bAQZLSut3Y1dWsyxQH0BYhmIdbFi3Q3ufvt',
        secure=False
    )

    bucket_name = 'kaggle-data'
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    data_path = '/tmp'
    for file_name in os.listdir(data_path):
        if file_name.endswith('.csv'):
            client.fput_object(bucket_name, file_name, os.path.join(data_path, file_name))

# Define the DAG
dag = DAG(
    'kaggle_to_minio',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Download data from Kaggle and upload to MinIO using MinIO Python client',
    schedule_interval=None,
    catchup=False
)

download_task = PythonOperator(
    task_id='download_kaggle_data',
    python_callable=download_kaggle_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_minio',
    python_callable=upload_to_minio,
    dag=dag,
)

download_task >> upload_task
