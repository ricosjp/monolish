FROM ghcr.io/ricosjp/allgebra/cuda10_2/mkl:20.10.0
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

RUN mkdir -p /opt/ricos/monolish/0.1/lib \
&&  mkdir -p /opt/ricos/monolish/0.1/include \
&&  mkdir -p /opt/ricos/monolish/0.1/samples 

COPY ./lib/ /opt/ricos/monolish/0.1/lib/
COPY ./include/ /opt/ricos/monolish/0.1/include/
COPY ./docker/monolish_sample/ /opt/ricos/monolish/0.1/samples

COPY ./lib/* /usr/lib64/
COPY ./include/* /usr/include/

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/ricos/monolish/0.1/lib/ 
ENV LIBRARY_PATH $LIBRARY_PATH:/opt/ricos/monolish/0.1/lib/ 
ENV CPATH $CPATH:/opt/ricos/monolish/0.1/include/
