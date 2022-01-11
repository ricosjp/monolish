# monolish: MONOlithic LInear equation Solvers for Highly-parallel architecture. {#mainpage}

## Quick start guide

monolish pre-build library and examples are composed in a container:

```
docker run -it --rm ghcr.io/ricosjp/monolish/mkl:${PROJECT_VERSION}
cd /usr/share/monolish/examples/blas/innerproduct
make cpu
```

There are also GPU-enabled images:

```
docker run -it --rm --gpus all ghcr.io/ricosjp/monolish/mkl-nvidia:${PROJECT_VERSION}
/usr/share/monolish/link_monolish_gpu.sh
cd /usr/share/monolish/examples/blas/innerproduct
make gpu
```

Be sure that you need to detect your GPU architecture (e.g. Ampare, Volta, ...) by `link_monolish_gpu.sh` script.

These compiles following example code:

@include blas/innerproduct/innerproduct.cpp

Please see [CPU Examples](@ref cpu_dev) and [GPU Examples](@ref gpu_dev) for more examples.

## For Users
- [What's monolish?](@ref monolish_intro)

### Installation
- [Installation using apt (for ubuntu 20.04)](@ref install_guide)
- [Docker container with monolish installed](@ref monolish_docker)

### Examples
- [CPU Examples](@ref cpu_dev)
- [GPU Examples](@ref gpu_dev)

### API Overview
- [**Modules**](./modules.html)
  - [Data types](@ref data_type)
  - [Supported Operations](@ref oplist)
  - [Linear Solvers](@ref solverlist)
  - [Performance logging and find bottlenecks](@ref logger)

## For Developers
- [Build monolish from source code](@ref build_guide)
- [Testing and Benchmarking](@ref test_bench)
- [Contribution guide](@ref contribution) 

## Citations
- [Publications](@ref publications)

## Links
- [**Source code**](https://github.com/ricosjp/monolish/)
- [**Documentation**](https://ricosjp.github.io/monolish/)
- [**Bug reports**](https://github.com/ricosjp/monolish/issues)
- [**Releases**](https://github.com/ricosjp/monolish/releases)
- [**Changelog**](https://github.com/ricosjp/monolish/blob/master/CHANGELOG.md)
- [**Licence**](https://github.com/ricosjp/monolish/blob/master/LICENSE)
- [**monolish log viewer**](https://pypi.org/project/monolish-log-viewer/)
- [**Continuous benchmarking**](https://ricosjp.github.io/monolish_benchmark_result/)

Copyright 2021 [RICOS Co. Ltd.](https://www.ricos.co.jp/)
