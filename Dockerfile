FROM quay.io/astronomer/astro-runtime:7.3.0

USER root

RUN apt-get update && \
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*

USER astro

RUN pip install --no-cache-dir apache-airflow-providers-google psycopg2-binary