all: cpu

cpu:
	make -j -C jacobi/ cpu
	make -j -C cg/ cpu
	make -j -C bicgstab/ cpu
	make -j -C dense_lu/ cpu
	make -j -C dense_cholesky/ cpu
	#make -j -C sparse_lu/ cpu
	#make -j -C sparse_qr/ cpu
	#make -j -C sparse_cholesky/ cpu

gpu:
	make -j -C jacobi/ gpu
	make -j -C cg/ gpu
	make -j -C bicgstab/ gpu
	make -j -C dense_lu/ gpu
	#make -j -C dense_cholesky/ gpu
	#make -j -C sparse_lu/ gpu
	make -j -C sparse_qr/ gpu
	make -j -C sparse_cholesky/ gpu

a64fx:
	make -j -C jacobi/ a64fx
	make -j -C cg/ a64fx
	make -j -C bicgstab/ a64fx

sxat:
	make -j -C jacobi/ sxat
	make -j -C cg/ sxat
	make -j -C bicgstab/ sxat

run_cpu:
	make -j -C jacobi/ run_cpu
	make -j -C cg/ run_cpu
	make -j -C bicgstab/ run_cpu
	make -j -C dense_lu/ run_cpu
	make -j -C dense_cholesky/ run_cpu
	# make -C sparse_lu/ run_cpu
	# make -j -C sparse_qr/ run_gpu
	#make -j -C sparse_cholesky/ run_cpu

run_gpu:
	make -j -C jacobi/ run_gpu
	make -j -C cg/ run_gpu
	make -j -C bicgstab/ run_gpu
	make -j -C dense_lu/ run_gpu
	# make -j -C dense_cholesky/ run_gpu
	# make -j -C sparse_lu/ run_gpu
	make -j -C sparse_qr/ run_gpu
	make -j -C sparse_cholesky/ run_gpu

run_a64fx:
	make -j -C jacobi/ run_a64fx
	make -j -C cg/ run_a64fx
	make -j -C bicgstab/ run_a64fx

run_sxat:
	make -j -C jacobi/ run_sxat
	make -j -C cg/ run_sxat
	make -j -C bicgstab/ run_sxat
	
clean:
	- make -j -C jacobi/ clean
	- make -j -C cg/ clean
	- make -j -C bicgstab/ clean
	- make -j -C dense_lu/ clean
	- make -j -C dense_cholesky/ clean
	- make -j -C sparse_lu/ clean
	- make -j -C sparse_qr/ clean
	- make -j -C sparse_cholesky/ clean
