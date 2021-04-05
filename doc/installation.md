# Installation using apt {#install_guide}

First, you need to enable CUDA 10.1 repository to enable cublas, cusolver, cusparse using following steps:
\code{shell}
$ sudo apt update && sudo apt install wget gnupg software-properties-common
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
\endcode

Download deb file from Releases page and install as follows:
\code{shell}
$ sudo apt update && sudo apt install wget
$ wget https://github.com/ricosjp/monolish/releases/download/0.14.0/monolish_0.14.0-1+oss+nvidia_amd64.deb
$ apt install ./monolish_0.14.0-1+oss+nvidia_amd64.deb
\endcode

- +oss+nvidia is OSS+NVIDIA variant
- +intel+nvidia is Intel+NVIDIA variant

cusolver, cusparse libraries dependencies are automatically resolved via apt.

## Other Archtectures or OS
see [here](@ref build_guide)

