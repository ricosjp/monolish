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
$ wget ${monolish_release_download_base}/${monolish_deb_common}
$ wget ${monolish_release_download_base}/${monolish_deb_oss}
$ sudo apt install -y ./${monolish_deb_common} ./${monolish_deb_oss}
\endcode

There are two variants according to backend BLAS and LAPACK implementation:

- [+oss][deb_oss] means it uses OpenBLAS
- [+mkl][deb_mkl] means it uses Intel MKL

## For NVIDIA GPU
First, you need to nable CUDA 11.4 repository to enable cuBLAS, cuSPARSE, cuSOLVER using following steps:e

\code{shell}
$ sudo apt install -y gnupg software-properties-common
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
$ sudo apt install -y cuda-11-4
\endcode

\code{shell}
$ wget ${monolish_release_download_base}/${monolish_deb_common}
$ wget ${monolish_release_download_base}/${monolish_deb_oss_nvidia}
$ sudo apt install -y ./${monolish_deb_common} ./${monolish_deb_oss_nvidia}
\endcode


[release]: ${monolish_release_url}
[deb_oss]: ${monolish_release_download_base}/${monolish_deb_oss}
[deb_mkl]: ${monolish_release_download_base}/${monolish_deb_mkl}
[deb_oss_nvidia]: ${monolish_release_download_base}/${monolish_deb_oss_nvidia}
[deb_mkl_nvidia]: ${monolish_release_download_base}/${monolish_deb_mkl_nvidia}
[deb_common]: ${monolish_release_download_base}/${monolish_deb_common}