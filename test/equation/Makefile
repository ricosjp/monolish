all: cpu

cpu:
	make -j -C cg/ cpu
	make -j -C bicgstab/ cpu
	#make -j -C sparse_lu/ cpu
	#make -j -C sparse_qr/ cpu
	#make -j -C sparse_cholesky/ cpu

gpu:
	make -j -C cg/ gpu
	make -j -C bicgstab/ gpu
	#make -j -C sparse_lu/ gpu
	make -j -C sparse_qr/ gpu
	make -j -C sparse_cholesky/ gpu

fx:
	make -j -C cg/ fx
	make -j -C bicgstab/ fx

sx:
	make -j -C cg/ sx
	make -j -C bicgstab/ sx

run_cpu:
	make -j -C cg/ run_cpu
	make -j -C bicgstab/ run_cpu
	# make -C sparse_lu/ run_cpu
	# make -j -C sparse_qr/ run_gpu
	#make -j -C sparse_cholesky/ run_cpu

run_gpu:
	make -j -C cg/ run_gpu
	make -j -C bicgstab/ run_gpu
	# make -j -C sparse_lu/ run_gpu
	make -j -C sparse_qr/ run_gpu
	make -j -C sparse_cholesky/ run_gpu

run_fx:
	make -j -C cg/ run_fx
	make -j -C bicgstab/ run_fx

run_sx:
	make -j -C cg/ run_sx
	make -j -C bicgstab/ run_sx
	
clean:
	- make -j -C cg/ clean
	- make -j -C bicgstab/ clean
	- make -j -C sparse_lu/ clean
	- make -j -C sparse_qr/ clean
	- make -j -C sparse_cholesky/ clean
