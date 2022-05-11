# Build monolish from source code {#build_guide}

# Introduction

The GPU implementation requires OpenMP Offloading in clang.
Enabling OpenMP Offloading requires an option to be specified when building clang. This is not practical.

To share the development environment, the monolish development environment is provided in an [allgebra Docker container](https://github.com/ricosjp/allgebra).

If you want to add a feature that depends on a new library, please make a pull request to allgebra.

## Enter to allgebra docker container and Building monolish

`cmake` and `docker` command options are complex. We provide shortcut commands for these commands in `Makefile`.
The options of cmake are listed at next section.

- CPU, MKL

```
git clone git@github.com:ricosjp/monolish.git
make in_mkl_cpu # shortcut command of docker
make install_cpu # shortcut command of cmake
```

- CPU, OSS (for AMD, ARM, Power)

```
git clone git@github.com:ricosjp/monolish.git
make in_oss_cpu
make install_cpu
```

- GPU, MKL

```
git clone git@github.com:ricosjp/monolish.git
make in_mkl_gpu
make install_gpu
```

- GPU, OSS (for NVIDIA GPU on AMD)

```
git clone git@github.com:ricosjp/monolish.git
make in_oss_gpu
make install_gpu
```

monolish will be installed in `MONOLISH_DIR`.
By default, monolish is installed in `/opt/monolish/lib/` and `/opt/monolish/include/`.

In the current version, it is necessary to specify the path to MONOLISH_DIR in order to compile the sample code. For example:

```
g++ -L/opt/monolish/lib/ -I /opt/monolish/include/ sample.cpp -lmonolish_cpu
```

## CMake build options

- **MONOLISH_USE_AVX**: _BOOL_ (defalut: ON)
  - Enable x86/64 SIMD AVX.
- **MONOLISH_CCACHE_BUILD**: _BOOL_ (default: ON)
  - Use ccache
- **MONOLISH_USE_MPI**: _BOOL_ (default: OFF)
  - Enable MPI.
  - MPI functions are under development. We do not recommend that users use this option.**
- **MONOLISH_PACKAGE_COMMON**: _BOOL_ (default:OFF)
  - Create libmonolish package, common contents of any backend
- **MONOLISH_USE_NVIDIA_GPU**: _BOOL_ (default: OFF)
  - Build with NVIDIA GPU.
  - This option automatically determines the NVIDIA GPU architecture and builds.
- **MONOLISH_NVIDIA_GPU_ARCH_ALL**: _BOOL_ (default OFF)
  - Build library for all known CUDA architectures.
  - We get "all known CUDA architecture" from LLVM's OpenMP Offloading support information.
- **MONOLISH_NVIDIA_GPU_ARCH**: _STRING_
  - Specify NVIDIA GPU architecture used while OpenMP Offloading. e.g. sm_52.
- **MONOLISH_PACKAGE_VERSION**: _STRING_
  - Specify Package version for deb and containers.
- **MONOLISH_CONTAINER_REGISTRY**: _STRING_ (default: `ghcr.io/ricosjp/monolish`)
  - Container registry where monolish container will be uploaded.
- **MONOLISH_CONTAINER_BASE_IMAGE**: _STRING_ (default: `ghcr.io/ricosjp/allgebra/cuda11_4/clang13/mkl`)
  - Base image upon which monolish container will be created.
- **MONOLISH_CONTAINER_BASE_TAG**: _STRING_ (default: `22.05.1`)
  - Tag of MONOLISH_CONTAINER_BASE_IMAGE

## Build monolish on the local for CPU (Ubuntu)

If Docker is not used, users need to install MKL, OpenBLAS, etc. which they depend on.

monolish requires:

- git
- make
- cmake 3.17 or higher
- g++ or clang++ with C++17 support
- MKL or OpenBLAS

### Install make, cmake, and compilers

Install cmake 3.18:

```
curl -LO https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-Linux-x86_64.tar.gz \ && tar xf cmake-3.18.4-Linux-x86_64.tar.gz \ && mv cmake-3.18.4-Linux-x86_64/bin/* /usr/bin/ \ && mv cmake-3.18.4-Linux-x86_64/share/cmake-3.18 /usr/share/ \ && rm -rf cmake-3.18.4-Linux-x86_64*

```

Install gcc, g++, make:

```
apt install -y make gcc g++
```

### Install BLAS Library

Install MKL

```
apt install -y libmkl-dev
```

or install OpenBLAS

```
apt install -y libopenblas-dev
```

### Install monolish

```
git clone git@github.com:ricosjp/monolish.git
cd monolish/
make install_cpu
```

Install directory is `MONOLIDH_DIR`.

## Build monolish on the local for GPU

Installation on the local environment is not supported. This is because monolish requires a clang with OpenMP Offloading enabled.

OpenMP Offloading enabled clang is provided by the following allgebra container.

[allgebra Docker container](https://github.com/ricosjp/allgebra).

## Build monolish for special hardware

SXAT and A64fx do not require Docker.

### NEC SX-Aurora TSUBASA

```
git clone git@github.com:ricosjp/monolish.git
make sxat
make install_sxat
```

We use `NEC nc++ 3.2.1` and `NEC NLC 2.3.0`.

VE requires the setting of an environment variable `VE_LD_LIBRARY_PATH`.

### Fujitsu A64FX

```
git clone git@github.com:ricosjp/monolish.git
make a64fx
make install_a64fx
```

monolish only support fcc clang mode.

## AMD Radeon GPU and Intel Xe GPU

Currently, it does not work because the code depends on NVIDIA CUDA libraries.

OpenMP Offloading should work on these architecture, so it should be possible to make it work based on the design concept of monolish.

These architectures will be supported in the future.
