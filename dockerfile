FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL maintainer "alvinprayuda@indexalaw.id"

WORKDIR /opt

RUN apt update
RUN apt install -y software-properties-common wget htop build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libbz2-dev

RUN wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz &&\
    tar -xf Python-3.7.3.tar.xz &&\
    cd Python-3.7.3 &&\
    ./configure --enable-optimizations &&\
    make -j8 &&\
    make altinstall

RUN pip3.7 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl &&\
    pip3.7 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl &&\
    pip3.7 install pytorch-transformers