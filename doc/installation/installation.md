# Installation using apt {#install_guide}

## For CPU

Download deb file from Releases page and install as follows:
\code{shell}
$ sudo apt update && sudo apt install wget
$ wget https://github.com/ricosjp/monolish/releases/download/0.14.2/monolish_0.14.2-1+oss_amd64.deb
$ sudo apt install ./monolish_0.14.2-1+oss_amd64.deb
\endcode

- +oss is `OSS` variant
- +mkl is `MKL` variant

## For GPU
First, you need to enable CUDA 10.1 repository to enable cuBLAS, cuSPARSE, cuSOLVER using following steps:
\code{shell}
$ sudo apt update && sudo apt install wget gnupg software-properties-common
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
$ sudo apt install libcublas10=10.2.1.243-1
\endcode

Download deb file from Releases page and install as follows:
\code{shell}
$ sudo apt update && sudo apt install wget
$ wget https://github.com/ricosjp/monolish/releases/download/0.14.2/monolish_0.14.2-1+oss+nvidia_amd64.deb
$ sudo apt install ./monolish_0.14.2-1+oss+nvidia_amd64.deb
\endcode

- +oss+nvidia is `OSS`+`NVIDIA` variant
- +mkl+nvidia is `MKL`+`NVIDIA` variant

cuSPARSE, cuSOLVER libraries dependencies are automatically resolved via apt.

## Other Archtectures or OS
See [here](@ref build_guide).
