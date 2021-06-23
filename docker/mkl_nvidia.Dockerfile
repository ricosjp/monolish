FROM ghcr.io/ricosjp/allgebra/cuda10_1/clang11gcc7/mkl:21.05.0

ENV DEBIAN_FRONTEND=noninteractive
COPY ./monolish_examples/ /opt/monolish/examples/
COPY ./monolish_benchmark/ /opt/monolish/benchmark/

RUN apt-get update -y \
 && apt-get install -y wget python3-pip \
 && apt-get clean && \
 pip3 install monolish-log-viewer==0.1.1

# install libmonolish_gpu.so
RUN wget https://github.com/ricosjp/monolish/releases/download/0.14.0/monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt install -y ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && rm ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./link_monolish_gpu.sh /opt/monolish/link_monolish_gpu.sh
RUN chmod a+x /opt/monolish/link_monolish_gpu.sh
