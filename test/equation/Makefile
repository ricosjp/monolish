all: cpu

cpu:
	make -j -C iterative/ cpu
	make -j -C dense_lu/ cpu
	make -j -C dense_cholesky/ cpu
	#make -j -C sparse_qr/ cpu
	#make -j -C sparse_cholesky/ cpu
	#make -j -C sparse_ilu/ cpu

gpu:
	make -j -C iterative/ gpu
	make -j -C dense_lu/ gpu
	#make -j -C dense_cholesky/ gpu
	make -j -C sparse_qr/ gpu
	make -j -C sparse_cholesky/ gpu
	make -j -C sparse_ilu/ gpu

a64fx:
	make -j -C iterative/ a64fx
	make -j -C dense_lu/ a64fx
	make -j -C dense_cholesky/ a64fx
	#make -j -C sparse_qr/ a64fx
	#make -j -C sparse_cholesky/ a64fx
	#make -j -C sparse_ilu/ a64fx

sxat:
	make -j -C iterative/ sxat
	make -j -C dense_lu/ sxat
	make -j -C dense_cholesky/ sxat
	#make -j -C sparse_qr/ sxat
	#make -j -C sparse_cholesky/ sxat
	#make -j -C sparse_ilu/ sxat

run_cpu:
	make -j -C iterative/ run_cpu
	make -j -C dense_lu/ run_cpu
	make -j -C dense_cholesky/ run_cpu
	#make -j -C sparse_qr/ run_cpu
	#make -j -C sparse_cholesky/ run_cpu
	#make -j -C sparse_ilu/ run_cpu

run_gpu:
	make -j -C iterative/ run_gpu
	make -j -C dense_lu/ run_gpu
	# make -j -C dense_cholesky/ run_gpu
	make -j -C sparse_qr/ run_gpu
	make -j -C sparse_cholesky/ run_gpu
	make -j -C sparse_ilu/ run_gpu

run_a64fx:
	make -j -C iterative/ run_a64fx
	make -j -C dense_lu/ run_a64fx
	make -j -C dense_cholesky/ run_a64fx
	#make -j -C sparse_qr/ run_a64fx
	#make -j -C sparse_cholesky/ run_a64fx
	#make -j -C sparse_ilu/ run_a64fx

run_sxat:
	make -j -C iterative/ run_sxat
	make -j -C dense_lu/ run_sxat
	make -j -C dense_cholesky/ run_sxat
	#make -j -C sparse_qr/ run_sxat
	#make -j -C sparse_cholesky/ run_sxat
	#make -j -C sparse_ilu/ run_sxat
	
clean:
	- make -j -C iterative/ clean
	- make -j -C dense_lu/ clean
	- make -j -C dense_cholesky/ clean
	- make -j -C sparse_qr/ clean
	- make -j -C sparse_cholesky/ clean
	- make -j -C sparse_ilu/ clean
