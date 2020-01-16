FROM registry.ritc.jp/ricos/allgebra/base
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

RUN mkdir -p /opt/ricos/monolish/0.1/lib \
&&  mkdir -p /opt/ricos/monolish/0.1/include \
&&  mkdir -p /monolish_sample/

COPY ./lib/ /opt/ricos/monolish/0.1/lib/
COPY ./include/ /opt/ricos/monolish/0.1/include/
COPY ./test/ /monolish_sample/

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/ricos/monolish/0.1/lib/
ENV LIBRARY_PATH $LIBRARY_PATH:/opt/ricos/monolish/0.1/lib/
ENV CPATH $CPATH:/opt/ricos/monolish/0.1/include/
