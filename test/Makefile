all:cpu

cpu:
	make -C common/vector/ cpu
	make -C common/matrix_common/ cpu
	make -C common/arithmetic_sv/ cpu
	make -C common/arithmetic_vv/ cpu
	#
	make -C blas/axpy/ cpu
	make -C blas/axpyz/ cpu
	make -C blas/xpay/ cpu
	make -C blas/scal/ cpu
	make -C blas/dot/ cpu
	make -C blas/nrm2/ cpu
	make -C blas/spmv/ cpu
	make -C blas/scal_crs/ cpu
	#
	make -C equation/cg/ cpu
	make -C equation/sparse_lu/ cpu
	#
	make -C logger/logging/ cpu

gpu:
	make -C common/vector/ gpu
	make -C common/matrix_common/ gpu
	make -C common/arithmetic_sv/ gpu
	make -C common/arithmetic_vv/ gpu
	#
	make -C blas/axpy/ gpu
	make -C blas/axpyz/ gpu
	make -C blas/xpay/ gpu
	make -C blas/scal/ gpu
	make -C blas/dot/ gpu
	make -C blas/nrm2/ gpu
	make -C blas/spmv/ gpu
	make -C blas/scal_crs/ gpu
	#
	make -C equation/cg/ gpu
	make -C equation/sparse_lu/ gpu
	#
	make -C logger/logging/ gpu

run_cpu:
	make -C common/vector/ run_cpu
	make -C common/matrix_common/ run_cpu
	make -C common/arithmetic_sv/ run_cpu
	make -C common/arithmetic_vv/ run_cpu
	#
	make -C blas/axpy/ run_cpu
	make -C blas/axpyz/ run_cpu
	make -C blas/xpay/ run_cpu
	make -C blas/scal/ run_cpu
	make -C blas/dot/ run_cpu
	make -C blas/nrm2/ run_cpu
	make -C blas/spmv/ run_cpu
	make -C blas/scal_crs/ run_cpu
	#
	make -C equation/cg/ run_cpu
# 	make -C equation/sparse_lu/ run_cpu
	#
	make -C logger/logging/ run_cpu

run_gpu:
	make -C common/vector/ run_gpu
	make -C common/matrix_common/ run_gpu
	make -C common/arithmetic_sv/ run_gpu
	make -C common/arithmetic_vv/ run_gpu
	#
	make -C blas/axpy/ run_gpu
	make -C blas/axpyz/ run_gpu
	make -C blas/xpay/ run_gpu
	make -C blas/scal/ run_gpu
	make -C blas/dot/ run_gpu
	make -C blas/nrm2/ run_gpu
	make -C blas/spmv/ run_gpu
	make -C blas/scal_crs/ run_gpu
	#
	make -C equation/sparse_lu/ run_gpu
# 	make -C equation/cg/ run_gpu
	#
	make -C logger/logging/ run_gpu

run_gpu_profile:
	make PROFILER=nvprof -C blas/dot/ run_gpu

clean:
	make -C common/vector/ clean
	make -C blas/dot/ clean
	make -C blas/spmv/ clean
	make -C equation/cg/ clean
	make -C equation/sparse_lu/ clean
