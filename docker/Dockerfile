FROM ghcr.io/ricosjp/allgebra/clang11gcc7/cuda10_1/mkl:latest
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

COPY . /monolish

ENV MONOLISH_DIR /opt/ricos/monolish/0.3.2

# Build and install libmonolish_gpu.so
RUN export CC=/usr/local/llvm-11.0.0/bin/clang \
 && export CC=/usr/local/llvm-11.0.0/bin/clang++ \
 && cmake -Bbuild -DMONOLISH_USE_GPU=ON -DCMAKE_INSTALL_PREFIX=${MONOLISH_DIR} /monolish \
 && cmake --build build --target install -- -j $(nproc) \
 && rm -rf build

# Build and install libmonolish_cpu.so
RUN cmake -Bbuild -DMONOLISH_USE_GPU=OFF -DCMAKE_INSTALL_PREFIX=${MONOLISH_DIR} /monolish \
 && cmake --build build --target install -- -j $(nproc) \
 && rm -rf build

RUN echo "${MONOLISH_DIR}/lib" > /etc/ld.so.conf.d/monolish.conf && ldconfig

ENV CPATH              ${MONOLISH_DIR}/include
ENV C_INCLUDE_PATH     ${MONOLISH_DIR}/include
ENV CPLUS_INCLUDE_PATH ${MONOLISH_DIR}/include
ENV LIBRARY_PATH       ${MONOLISH_DIR}/lib
