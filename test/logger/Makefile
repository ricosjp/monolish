all: cpu

cpu:
	make -j -C logging/ cpu

gpu:
	make -j -C logging/ gpu

run_cpu:
	make -j -C logging/ run_cpu

run_gpu:
	make -j -C logging/ run_gpu
	
clean:
	- make -j -C logging/ clean
