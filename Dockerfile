FROM nvcr.io/nvidia/pytorch:22.12-py3

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app