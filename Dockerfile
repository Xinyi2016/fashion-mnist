FROM tensorflow/tensorflow:1.12.0-gpu-py3

MAINTAINER xinyi.zeng@connect.polyu.hk

WORKDIR /

# Install necessary packages.
RUN apt-get -y update && \
    apt-get -y install jq awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt

ADD . /

#ENTRYPOINT python ./app.py $ARGUMENTS
ENTRYPOINT python ./benchmark/runner.py