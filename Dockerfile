FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV TZ Europe/Moscow
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 LANGUAGE=en_US:en

RUN apt update --fix-missing && \
    apt install -y wget bzip2 ca-certificates libglib2.0-0  libgl1-mesa-glx build-essential

COPY . .

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3", "validation.py"]
