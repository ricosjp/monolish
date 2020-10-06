FROM ghcr.io/ricosjp/allgebra/cuda10_2/mkl:20.10.0
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

COPY . /monolish

ENV MONOLISH_DIR /opt/ricos/monolish/0.3.2

RUN cmake -Bbuild -DMONOLISH_USE_GPU=ON -DCMAKE_INSTALL_PREFIX=${MONOLISH_DIR} /monolish \
 && cmake --build build --target install -- -j $(nproc) \
 && rm -rf build

RUN echo "${MONOLISH_DIR}/lib" > /etc/ld.so.conf.d/monolish.conf && ldconfig

ENV CPATH              ${MONOLISH_DIR}/include
ENV C_INCLUDE_PATH     ${MONOLISH_DIR}/include
ENV CPLUS_INCLUDE_PATH ${MONOLISH_DIR}/include
ENV LIBRARY_PATH       ${MONOLISH_DIR}/lib
