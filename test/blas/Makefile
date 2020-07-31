all: cpu 

cpu:
	make -C axpy/ cpu
	make -C axpyz/ cpu
	make -C xpay/ cpu
	make -C scal/ cpu
	make -C dot/ cpu
	make -C nrm2/ cpu
	make -C scal_crs/ cpu
	make -C scal_dense/ cpu
	make -C matvec_crs/ cpu
	make -C matvec_dense/ cpu
	make -C matmul_dense/ cpu

gpu:
	make -C axpy/ gpu
	make -C axpyz/ gpu
	make -C xpay/ gpu
	make -C scal/ gpu
	make -C dot/ gpu
	make -C nrm2/ gpu
	make -C scal_crs/ gpu
	make -C scal_dense/ gpu
	make -C matvec_crs/ gpu
	make -C matvec_dense/ gpu
	make -C matmul_dense/ gpu

run_cpu:
	make -C axpy/ run_cpu
	make -C axpyz/ run_cpu
	make -C xpay/ run_cpu
	make -C scal/ run_cpu
	make -C dot/ run_cpu
	make -C nrm2/ run_cpu
	make -C scal_crs/ run_cpu
	make -C scal_dense/ run_cpu
	make -C matvec_crs/ run_cpu
	make -C matvec_dense/ run_cpu
	make -C matmul_dense/ run_cpu

run_gpu:
	make -C axpy/ run_gpu
	make -C axpyz/ run_gpu
	make -C xpay/ run_gpu
	make -C scal/ run_gpu
	make -C dot/ run_gpu
	make -C nrm2/ run_gpu
	make -C scal_crs/ run_gpu
	make -C scal_dense/ run_gpu
	make -C matvec_crs/ run_gpu
	make -C matvec_dense/ run_gpu
	make -C matmul_dense/ run_gpu

clean:
	- make -C axpy/ clean
	- make -C axpyz/ clean
	- make -C xpay/ clean
	- make -C scal/ clean
	- make -C dot/ clean
	- make -C nrm2/ clean
	- make -C scal_crs/ clean
	- make -C scal_dense/ clean
	- make -C matvec_crs/ clean
	- make -C matvec_dense/ clean
	- make -C matmul_dense/ clean
