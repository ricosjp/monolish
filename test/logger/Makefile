all: cpu

cpu:
	make -j -C logging/ cpu

gpu:
	make -j -C logging/ gpu

fx:
	make -j -C logging/ fx

sx:
	make -j -C logging/ sx

run_cpu:
	make -j -C logging/ run_cpu

run_gpu:
	make -j -C logging/ run_gpu

run_fx:
	make -j -C logging/ run_fx

run_sx:
	make -j -C logging/ run_sx
	
clean:
	- make -j -C logging/ clean
