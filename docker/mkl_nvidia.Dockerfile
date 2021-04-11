FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
COPY examples/ /opt/monolish/examples/

RUN apt-get update -y \
 && apt-get install -y wget python3-pip \
 && apt-get clean && \
 pip3 install monolish-log-viewer==0.1.0

# install CUDA, cuBLAS, cuSPARSE, cuSOLVER
RUN apt-get install -y wget gnupg software-properties-common \
&& wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
&& mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
&& add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
&& apt-get install -y libcublas10=10.2.1.243-1

# install libmonolish_gpu.so
RUN wget https://github.com/ricosjp/monolish/releases/download/0.14.0/monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt install -y ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && rm ./monolish_0.14.0-1+mkl+nvidia_amd64.deb \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./util/allgebra_get_device_cc /bin/
COPY ./util/link_monolish_gpu.sh usr/bin/link_monolish_gpu.sh

RUN chmod a+x usr/bin/link_monolish_gpu.sh

CMD ["sh", "-c", "/bin/link_monolish_gpu.sh"]
