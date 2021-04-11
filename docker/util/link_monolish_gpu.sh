CC=`allgebra_get_device_cc`
rm -f /usr/lib/libmonolish_gpu.so
ln -s /usr/lib/libmonolish_gpu_$CC.so /usr/lib/libmonolish_gpu.so

/bin/bash
