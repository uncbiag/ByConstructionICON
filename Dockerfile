FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3

WORKDIR /oncoreg

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY utils ./utils
COPY test.sh .
COPY train.sh .
COPY inference_constricon.py .
COPY train_constricon_supervised.py .
COPY multiscale_constr_model.py .
COPY network_definition.py .

RUN chmod +x test.sh
RUN chmod +x train.sh
