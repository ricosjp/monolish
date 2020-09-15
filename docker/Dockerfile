FROM registry.ritc.jp/ricos/allgebra/cuda10_2:0.2.0
LABEL maintainer "Toshiaki Hishinuma <hishinuma.toshiaki@gmail.com>"

COPY . /monolish

RUN cmake -Bbuild -DBUILD_GPU=ON -DCMAKE_INSTALL_PREFIX=/usr /monolish \
 && cmake --build build --target install -- -j $(nproc) \
 && rm -rf build
