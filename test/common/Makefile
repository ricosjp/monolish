all:cpu

cpu:
	make -C vector/ cpu
	make -C matrix_common/ cpu
	make -C arithmetic_sv/ cpu
	make -C arithmetic_vv/ cpu
	make -C convert/ cpu
	make -C transpose/ cpu

gpu:
	make -C vector/ gpu
	make -C matrix_common/ gpu
	make -C arithmetic_sv/ gpu
	make -C arithmetic_vv/ gpu
	make -C convert/ gpu
	make -C transpose/ gpu

fx:
	make -C vector/ fx
	make -C matrix_common/ fx
	make -C arithmetic_sv/ fx
	make -C arithmetic_vv/ fx
	make -C convert/ fx
	make -C transpose/ fx

run_cpu:
	make -C vector/ run_cpu
	make -C matrix_common/ run_cpu
	make -C arithmetic_sv/ run_cpu
	make -C arithmetic_vv/ run_cpu
	make -C convert/ run_cpu
	make -C transpose/ run_cpu

run_gpu:
	make -C vector/ run_gpu
	make -C matrix_common/ run_gpu
	make -C arithmetic_sv/ run_gpu
	make -C arithmetic_vv/ run_gpu
	make -C convert/ run_gpu
	make -C transpose/ run_gpu

run_fx:
	make -C vector/ run_fx
	make -C matrix_common/ run_fx
	make -C arithmetic_sv/ run_fx
	make -C arithmetic_vv/ run_fx
	make -C convert/ run_fx
	make -C transpose/ run_fx

clean:
	- make -C vector/ clean
	- make -C matrix_common/ clean
	- make -C arithmetic_sv/ clean
	- make -C arithmetic_vv/ clean
	make -C convert/ clean
	make -C transpose/ clean
