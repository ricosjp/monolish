# Installation using apt {#install_guide}

This page describes how to install pre-build monolish onto Ubuntu 22.04 LTS.
Following commands assumes `wget` command exists. It can be installed by

\code{shell}
$ sudo apt update -y
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
First, you need to nable CUDA 11.4 repository to enable cuBLAS, cuSPARSE, cuSOLVER using following steps:

\code{shell}
$ sudo apt install -y gnupg software-properties-common
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863ccpub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
$ sudo apt install -y cuda-11-7
\endcode

Then, install monolish by following steps:

\code{shell}
$ wget ${monolish_release_download_base}/${monolish_deb_common}
$ wget ${monolish_release_download_base}/${monolish_deb_oss_nvidia}
$ sudo apt install -y ./${monolish_deb_common} ./${monolish_deb_oss_nvidia}
\endcode

monolish for GPU has shared libraries for each generation of GPU.
Following commands set the path to the shared library for generation of GPU #0.

\code{shell}
$ export PATH=$PATH:/usr/local/cuda-11.4/bin/
$ /usr/share/monolish/link_monolish_gpu.sh
\endcode


[release]: ${monolish_release_url}
[deb_oss]: ${monolish_release_download_base}/${monolish_deb_oss}
[deb_mkl]: ${monolish_release_download_base}/${monolish_deb_mkl}
[deb_oss_nvidia]: ${monolish_release_download_base}/${monolish_deb_oss_nvidia}
[deb_mkl_nvidia]: ${monolish_release_download_base}/${monolish_deb_mkl_nvidia}
[deb_common]: ${monolish_release_download_base}/${monolish_deb_common}
