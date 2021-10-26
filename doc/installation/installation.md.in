# Installation using apt {#install_guide}

This page describes how to install pre-build monolish onto Ubuntu 20.04 LTS.
Following commands assumes `wget` command exists. It can be installed by

\code{shell}
$ sudo apt update
$ sudo apt install -y wget
\endcode

## For CPU

monolish deb file can be downloaded from [GitHub Release page][release]:

\code{shell}
$ wget ${monolish_release_download_base}/${monolish_deb_oss}
$ sudo apt install -y ./${monolish_deb_oss}
\endcode

There are two variants according to backend BLAS and LAPACK implementation:

- [+oss][deb_oss] means it uses OpenBLAS
- [+mkl][deb_mkl] means it uses Intel MKL

## For NVIDIA GPU

In the current version, monolish for GPU cannot be installed with apt.

Please use the container with [docker container with monolish installed](@ref monolish_docker).

See [here](@ref build_guide).

[release]: ${monolish_release_url}
[deb_oss]: ${monolish_release_download_base}/${monolish_deb_oss}
[deb_mkl]: ${monolish_release_download_base}/${monolish_deb_mkl}
[deb_oss_nvidia]: ${monolish_release_download_base}/${monolish_deb_oss_nvidia}
[deb_mkl_nvidia]: ${monolish_release_download_base}/${monolish_deb_mkl_nvidia}
