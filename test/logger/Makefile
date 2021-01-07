all: cpu

cpu:
	make -j -C logging/ cpu

gpu:
	make -j -C logging/ gpu

a64fx:
	make -j -C logging/ a64fx

sxat:
	make -j -C logging/ sxat

run_cpu:
	make -j -C logging/ run_cpu

run_gpu:
	make -j -C logging/ run_gpu

run_a64fx:
	make -j -C logging/ run_a64fx

run_sxat:
	make -j -C logging/ run_sxat
	
clean:
	- make -j -C logging/ clean
