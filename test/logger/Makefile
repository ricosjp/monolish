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
	make -j -C logging/ log_cpu
	# apt update -y
	# apt install -y python3-pandas

run_gpu:
	make -j -C logging/ run_gpu
	make -j -C logging/ log_gpu
	# apt update -y
	# apt install -y python3-pandas

run_fx:
	make -j -C logging/ run_fx
	make -j -C logging/ log_fx

run_sx:
	make -j -C logging/ run_sx
	make -j -C logging/ log_sx
	
clean:
	- make -j -C logging/ clean
