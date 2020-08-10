all:cpu

cpu:
	make -C common/ cpu
	make -C blas/ cpu
	make -C equation// cpu
	make -C logger/ cpu
	make -C lang/ cpu

gpu:
	make -C common/ gpu
	make -C blas/ gpu
	make -C equation/ gpu
	make -C logger/ gpu
	make -C lang/ gpu

fx:
	make -C common/ fx
	make -C blas/ fx
	make -C equation/ fx
	make -C logger/ fx
	make -C lang/ fx

run_cpu:
	make -C common/ run_cpu
	make -C blas/ run_cpu
	make -C equation/ run_cpu
# 	make -C equation/sparse_lu/ run_cpu
	make -C logger/ run_cpu
#	make -C lang/ run_cpu

run_gpu:
	make -C common/ run_gpu
	make -C blas/ run_gpu
	make -C equation/ run_gpu
	make -C logger/ run_gpu
	make -C lang/ run_gpu

run_fx:
	make -C common/ run_fx
	make -C blas/ run_fx
	make -C equation/ run_fx
	make -C logger/ run_fx
	make -C lang/ run_fx

run_gpu_profile:
	make PROFILER="nsys nvprof" -C common/ run_gpu
	make PROFILER="nsys nvprof" -C blas/ run_gpu
	make PROFILER="nsys nvprof" -C equation/ run_gpu
	make PROFILER="nsys nvprof" -C logger/ run_gpu
	make PROFILER="nsys nvprof" -C lang/ run_gpu

clean:
	- make clean -C common/ run_gpu
	- make clean -C blas/ run_gpu
	- make clean -C equation/ run_gpu
	- make clean -C logger/ run_gpu
