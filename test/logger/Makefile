all: cpu

cpu:
	make -j -C logging/ cpu

gpu:
	make -j -C logging/ gpu

run_cpu:
	make -j -C logging/ run_cpu
#	make -j -C logging/ log_cpu

run_gpu:
	make -j -C logging/ run_gpu
#	make -j -C logging/ log_gpu
	
clean:
	- make -j -C logging/ clean
