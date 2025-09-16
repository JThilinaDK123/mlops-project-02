FROM astrocrpublic.azurecr.io/runtime:3.0-10

RUN apt-get update && \
    apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env"

# Install apache-airflow-providers-google
RUN pip install --no-cache-dir apache-airflow-providers-google