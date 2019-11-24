all:cpu

cpu:
	make -C blas/dot/
	make -C blas/spmv/
	make -C equation/cg/
	make -C equation/sparse_lu/

gpu:
	make -C blas/dot/
	make -C blas/spmv/
	make -C equation/cg/
	make -C equation/sparse_lu/

run_cpu:
	make -C blas/dot/ run_cpu
	make -C blas/spmv/ run_cpu
	make -C equation/cg/ run_cpu
	make -C equation/sparse_lu/ run_cpu

clean:
	make -C blas/dot/ clean
	make -C blas/spmv/ clean
	make -C equation/cg/ clean
	make -C equation/sparse_lu/ clean
