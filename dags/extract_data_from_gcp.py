from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base_hook import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy

## Transform Step
def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  ## This should be the connection ID name relates to postgres connection in airflow
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@testing-3_607202-postgres-1:{conn.port}/{conn.schema}")
    ## "testing-3_607202-postgres-1", This part should replace by the container name
    ## Container name (postgres) can be found in docker desktop.
    df = pd.read_csv(file_path)
    df.to_sql(name = "titanic", con = engine, if_exists = "replace", index = False)

## Define the DAG
with DAG(
    dag_id = "extract_titanic_data", ## we can give any name here
    schedule_interval = None, 
    start_date = datetime(2023, 1, 1),
    catchup = False,
) as dag:

    ## Extract Step
    list_files = GCSListObjectsOperator(
        task_id = "list_files",
        bucket = "mlops_thilina_part02", ## Give the GCP bucket Name
    )


    download_file = GCSToLocalFilesystemOperator(
        task_id = "download_file",
        bucket = "mlops_thilina_part02", 
        object_name = "Titanic-Dataset.csv", ## Give the File Name
        filename = "/tmp/Titanic-Dataset.csv", 
    )


    ## Transforming and Loading Steps
    load_data = PythonOperator(
        task_id = "load_to_sql",
        python_callable = load_to_sql,
        op_kwargs = {"file_path": "/tmp/Titanic-Dataset.csv"}
    )

    list_files >> download_file >> load_data ## This is the workflow