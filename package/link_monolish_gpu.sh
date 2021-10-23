CC=`allgebra_get_device_cc`

rm -f /usr/lib/libmonolish_gpu.so
ln -s /usr/lib/libmonolish_gpu_$CC.so /usr/lib/libmonolish_gpu.so

nvidia-smi -L
echo "Compute Capability of GPU 0 is $CC"
echo "set /usr/lib/libmonolish_gpu.so -> /usr/lib/libmonolish_gpu_$CC.so"
