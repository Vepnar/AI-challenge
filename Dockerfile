FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tmp/

COPY requirements.txt .

RUN python -m pip install -r requirements.txt
USER 1000:1000
RUN nvidia-smi


CMD python training.py