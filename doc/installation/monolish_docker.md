# monolish container {#monolish_docker}

monolish container is a container for easy use of monolish based on ubuntu.

monolish is installed in the following locations:

- /usr/lib/
- /usr/include/

examples can be found at:

- /usr/share/monolish/examples/

benchmarks can be found at:

- /usr/share/monolish/benchmark/

# Run monolish container

Use the following command to enter the monolish docker container.
Please refer to the next section to see how to run the sample.

## OSS

```
docker run -it --rm ghcr.io/ricosjp/monolish/oss:${PROJECT_VERSION}
```

## MKL

```
docker run -it --rm ghcr.io/ricosjp/monolish/mkl:${PROJECT_VERSION}
```

## OSS+NVIDIA

If there is not a GPU, `--gpus all` will fail.

```
docker run -it --rm --gpus all ghcr.io/ricosjp/monolish/oss-nvidia:${PROJECT_VERSION}
```

Then, in execute the following command in monolish docker.
The monolish library has `so` files for each compute capability (CC) of GPU.
This script determines the CC of GPU 0 and makes a link to `libmonolish_gpu.so`.

```
/usr/share/monolish/link_monolish_gpu.sh
```

## MKL+NVIDIA

If there is not a GPU, `--gpus all` will fail.

```
docker run -it --rm --gpus all ghcr.io/ricosjp/monolish/mkl-nvidia:${PROJECT_VERSION}
```

Then, in execute the following command in monolish docker.
The monolish library has `so` files for each compute capability (CC) of GPU.
This script determines the CC of GPU 0 and makes a link to `libmonolish_gpu.so`.

```
/usr/share/monolish/link_monolish_gpu.sh
```

# Build monolish container (for developers)

In `monolish/docker` :

## OSS

```
make oss
```

## MKL

```
make mkl
```

## OSS+NVIDIA

```
make oss_nvidia
```

## MKL+NVIDIA

```
make mkl_nvidia
```

# Testing monolish docker

```
make test
```
