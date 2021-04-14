# monolish container
ubuntu 20.04 with monolish installed

- Ubuntu 20.04
- gcc 9.3.0

monolish is installed in the following locations:
- /usr/lib/
- /usr/include

examples can be fount at:
- /opt/monolish/

# Build monolish container

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
make in_oss_nvidia
```

## MKL+NVIDIA

```
make mkl_nvidia
```

# Enter monolish container

## OSS

```
make in_oss
```

## MKL

```
make in_mkl
```

## OSS+NVIDIA

```
make oss_nvidia
```

## MKL+NVIDIA

```
make in_mkl_nvidia
```
