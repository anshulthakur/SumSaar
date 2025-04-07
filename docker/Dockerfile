FROM apache/airflow:2.10.5-python3.12
USER root
RUN apt update
RUN apt-get install python3 python3-dev libxml2-dev libxslt-dev -y
RUN apt-get install libjpeg-dev zlib1g-dev libpng-dev -y
USER airflow
ADD requirements.txt .
RUN pip install -r requirements.txt