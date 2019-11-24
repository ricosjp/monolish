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

clean:
	make -C blas/dot/ clean
	make -C blas/spmv/ clean
	make -C equation/cg/ clean
	make -C equation/sparse_lu/ clean
