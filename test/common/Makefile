all:cpu

cpu:
	make -C vector/ cpu
	make -C matrix_common/ cpu
	make -C arithmetic_sv/ cpu
	make -C arithmetic_vv/ cpu

gpu:
	make -C vector/ gpu
	make -C matrix_common/ gpu
	make -C arithmetic_sv/ gpu
	make -C arithmetic_vv/ gpu

run_cpu:
	make -C vector/ run_cpu
	make -C matrix_common/ run_cpu
	make -C arithmetic_sv/ run_cpu
	make -C arithmetic_vv/ run_cpu

run_gpu:
	make -C vector/ run_gpu
	make -C matrix_common/ run_gpu
	make -C arithmetic_sv/ run_gpu
	make -C arithmetic_vv/ run_gpu

clean:
	- make -C vector/ clean
	- make -C matrix_common/ clean
	- make -C arithmetic_sv/ clean
	- make -C arithmetic_vv/ clean
