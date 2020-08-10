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

fx:
	make -C vector/ fx
	make -C matrix_common/ fx
	make -C arithmetic_sv/ fx
	make -C arithmetic_vv/ fx

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

run_fx:
	make -C vector/ run_fx
	make -C matrix_common/ run_fx
	make -C arithmetic_sv/ run_fx
	make -C arithmetic_vv/ run_fx

clean:
	- make -C vector/ clean
	- make -C matrix_common/ clean
	- make -C arithmetic_sv/ clean
	- make -C arithmetic_vv/ clean
