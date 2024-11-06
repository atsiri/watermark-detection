python3 -m venv airflow-venv 

source airflow-venv/bin/activate

pip3 install apache-airflow 

airflow db init
or
airflow db migrate

airflow users list

airflow users create --username admin --password admin --firstname airflow --lastname airflow --role Admin --email admin@admin.com

airflow webserver -p 8080 & airflow scheduler &

#docker compose
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.2/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins ./config
export AIRFLOW_UID=50000
