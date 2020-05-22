all: cpu

cpu:
	make -j -C cg/ cpu
	#make -j -C sparse_lu/ cpu
	#make -j -C sparse_qr/ cpu

gpu:
	make -j -C cg/ gpu
	#make -j -C sparse_lu/ gpu
	make -j -C sparse_qr/ gpu

run_cpu:
	make -j -C cg/ run_cpu
	# make -C sparse_lu/ run_cpu
	# make -j -C sparse_qr/ run_gpu

run_gpu:
	make -j -C cg/ run_gpu
	# make -j -C sparse_lu/ run_gpu
	make -j -C sparse_qr/ run_gpu
	
clean:
	- make -j -C cg/ clean
	- make -j -C sparse_lu/ clean
	- make -j -C sparse_qr/ clean
