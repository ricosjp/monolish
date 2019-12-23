all:cpu

cpu:
	make -C blas/dot/ cpu
	make -C blas/spmv/ cpu
	make -C equation/cg/ cpu
	make -C equation/sparse_lu/ cpu

gpu:
	make -C blas/dot/ gpu
	make -C blas/spmv/ gpu
	make -C equation/cg/ gpu
	make -C equation/sparse_lu/ gpu

run_cpu:
	make -C blas/dot/ run_cpu
	make -C blas/spmv/ run_cpu
# 	make -C equation/cg/ run_cpu
# 	make -C equation/sparse_lu/ run_cpu

run_gpu:
	make -C blas/dot/ run_gpu
	make -C blas/spmv/ run_gpu
	make -C equation/sparse_lu/ run_gpu
# 	make -C equation/cg/ run_gpu
#
run_gpu_profile:
	make PROFILER=nvprof -C blas/dot/ run_gpu

clean:
	make -C blas/dot/ clean
	make -C blas/spmv/ clean
	make -C equation/cg/ clean
	make -C equation/sparse_lu/ clean
