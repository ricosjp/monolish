all: cpu

cpu: 
	make -C ./blas/cg_impl/ cpu
	make -C ./blas/innerproduct/ cpu
	make -C ./equation/cg/ cpu
	make -C ./equation/templated_solver/ cpu
	make -C ./eigen/DC/ cpu
	make -C ./eigen/lobpcg/ cpu

gpu: 
	make -C ./blas/cg_impl/ gpu
	make -C ./blas/innerproduct/ gpu
	make -C ./equation/cg/ gpu
	make -C ./equation/templated_solver/ gpu
	make -C ./eigen/DC/ gpu
	make -C ./eigen/lobpcg/ gpu

clean: 
	make -C ./blas/cg_impl/ clean
	make -C ./blas/innerproduct/ clean
	make -C ./equation/cg/ clean
	make -C ./equation/templated_solver/ clean
	make -C ./eigen/DC/ clean
	make -C ./eigen/lobpcg/ clean
