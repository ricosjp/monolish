#!/bin/bash
set -eu

gpu_cc=$(nvcc -o /usr/share/monolish/get_device_cc --run --run-args 0 /usr/share/monolish/get_device_cc.cu)
nvidia-smi -L
echo -e "Compute Capability of GPU 0 is ${gpu_cc}"

update-alternatives --set monolish /usr/lib/libmonolish_gpu_${gpu_cc}.so 
