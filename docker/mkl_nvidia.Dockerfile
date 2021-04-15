FROM ghcr.io/ricosjp/allgebra/cuda10_1/clang11gcc7/mkl

ENV DEBIAN_FRONTEND=noninteractive
COPY ./monolish_examples/ /opt/monolish/examples/

RUN apt-get update -y \
 && apt-get install -y wget python3-pip \
 && apt-get clean && \
 pip3 install monolish-log-viewer==0.1.0

# install libmonolish_gpu.so
RUN wget https://github.com/ricosjp/monolish/releases/download/0.14.0/monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt install -y ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && rm ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./util/allgebra_get_device_cc /bin/
COPY ./util/link_monolish_gpu.sh /usr/bin/link_monolish_gpu.sh

RUN chmod a+x usr/bin/link_monolish_gpu.sh

CMD ["sh", "-c", "/bin/link_monolish_gpu.sh"]
