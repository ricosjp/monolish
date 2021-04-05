# Build monolish from source code {#build_guide}

# Introduction

The GPU implementation requires OpenMP Offloading in clang.
Enabling OpenMP Offloading requires an option to be specified when building clang. This is not practical.

To share the development environment, the monolish development environment is provided in an [allgebra Docker container](https://github.com/ricosjp/allgebra).

If you want to add a feature that depends on a new library, please make a pull request to allgebra.

## Enter to allgebra docker container and Building monolish

- CPU, MKL
```
git clone git@github.com:ricosjp/monolish.git
make in_mkl_cpu
make install
```

- CPU, OSS (for AMD, ARM, Power)
```
git clone git@github.com:ricosjp/monolish.git
make in_oss_cpu
make install
```

- GPU, MKL
```
git clone git@github.com:ricosjp/monolish.git
make in_mkl_gpu
make install
```

- GPU, OSS (for NVIDIA GPU on AMD)
```
git clone git@github.com:ricosjp/monolish.git
make in_mkl_cpu
make install
```

If Docker is not used, users need to install MKL, OpenBLAS, etc. which they depend on.

## Build monolish for special hardware
- NEC SX-Aurora TSUBASA
```
git clone git@github.com:ricosjp/monolish.git
make -f Makefile.sxat
```

- Fujitsu A64fx
```
git clone git@github.com:ricosjp/monolish.git
make -f Makefile.a64fx
```
SXAT and A64fx do not require Docker.


## AMD Radeon GPU and Intel Xe GPU
Currently, it does not work because the code depends on NVIDIA CUDA libraries.

OpenMP Offloading should work on these architecture, so it should be possible to make it work based on the design concept of monolish.

These architectures will be supported in the future.
