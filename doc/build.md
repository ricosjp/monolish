# Build monolish {#build_guide}

# Introduction

The GPU implementation requires OpenMP Offloading in clang.
Enabling OpenMP Offloading requires an option to be specified when building clang. This is not practical.

To share the development environment, the monolish development environment is provided in an [allgebra Docker container](https://github.com/ricosjp/allgebra).

If you want to add a feature that depends on a new library, please make a pull request to allgebra.

## Enter to allgebra docker container

- CPU, MKL
```
make in_mkl_cpu
```

- CPU, OSS
```
make in_oss_cpu
```

- GPU, MKL
```
make in_mkl_gpu
```

- GPU, OSS
```
make in_mkl_cpu
```

## Build on allgebra
## CPU 
```
> make cpu
> make install
```

### GPU 
```
> make gpu
> make install
```
