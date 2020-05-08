all:cpu

cpu:
	make -C common/ cpu
	make -C blas/ cpu
	make -C equation// cpu
	make -C logger/ cpu

gpu:
	make -C common/ gpu
	make -C blas/ gpu
	make -C equation/ gpu
	make -C logger/ gpu

run_cpu:
	make -C common/ run_cpu
	make -C blas/ run_cpu
	make -C equation/ run_cpu
# 	make -C equation/sparse_lu/ run_cpu
	make -C logger/ run_cpu

run_gpu:
	make -C common/ run_gpu
	make -C blas/ run_gpu
	make -C equation/ run_gpu
	make -C logger/ run_gpu

run_gpu_profile:
	make PROFILER=nvprof -C common/ run_gpu
	make PROFILER=nvprof -C blas/ run_gpu
	make PROFILER=nvprof -C equation/ run_gpu
	make PROFILER=nvprof -C logger/ run_gpu

clean:
	make clean -C common/ run_gpu
	make clean -C blas/ run_gpu
	make clean -C equation/ run_gpu
	make clean -C logger/ run_gpu
