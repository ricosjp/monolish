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
	make -C matmul_dense-dense/ cpu
	make -C matmul_crs-dense/ cpu

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
	make -C matmul_dense-dense/ gpu
	make -C matmul_crs-dense/ gpu

fx:
	make -C axpy/ fx
	make -C axpyz/ fx
	make -C xpay/ fx
	make -C scal/ fx
	make -C dot/ fx
	make -C nrm2/ fx
	make -C scal_crs/ fx
	make -C scal_dense/ fx
	make -C matvec_crs/ fx
	make -C matvec_dense/ fx
	make -C matmul_dense-dense/ fx
	make -C matmul_crs-dense/ fx

sx:
	make -C axpy/ sx
	make -C axpyz/ sx
	make -C xpay/ sx
	make -C scal/ sx
	make -C dot/ sx
	make -C nrm2/ sx
	make -C scal_crs/ sx
	make -C scal_dense/ sx
	make -C matvec_crs/ sx
	make -C matvec_dense/ sx
	make -C matmul_dense-dense/ sx
	make -C matmul_crs-dense/ sx

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
	make -C matmul_dense-dense/ run_cpu
	make -C matmul_crs-dense/ run_cpu

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
	make -C matmul_dense-dense/ run_gpu
	make -C matmul_crs-dense/ run_gpu

run_fx:
	make -C axpy/ run_fx
	make -C axpyz/ run_fx
	make -C xpay/ run_fx
	make -C scal/ run_fx
	make -C dot/ run_fx
	make -C nrm2/ run_fx
	make -C scal_crs/ run_fx
	make -C scal_dense/ run_fx
	make -C matvec_crs/ run_fx
	make -C matvec_dense/ run_fx
	make -C matmul_dense-dense/ run_fx
	make -C matmul_crs-dense/ run_fx

run_sx:
	make -C axpy/ run_sx
	make -C axpyz/ run_sx
	make -C xpay/ run_sx
	make -C scal/ run_sx
	make -C dot/ run_sx
	make -C nrm2/ run_sx
	make -C scal_crs/ run_sx
	make -C scal_dense/ run_sx
	make -C matvec_crs/ run_sx
	make -C matvec_dense/ run_sx
	make -C matmul_dense-dense/ run_sx
	make -C matmul_crs-dense/ run_sx

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
	- make -C matmul_dense-dense/ clean
	- make -C matmul_crs-dense/ clean
