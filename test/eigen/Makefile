all: cpu

cpu:
	make -j -C LOBPCG/ cpu

gpu:
	make -j -C LOBPCG/ gpu

a64fx:
	make -j -C LOBPCG/ a64fx

sxat:
	make -j -C LOBPCG/ sxat

run_cpu:
	make -j -C LOBPCG/ run_cpu

run_gpu:
	make -j -C LOBPCG/ run_gpu

run_a64fx:
	make -j -C LOBPCG/ run_a64fx

run_sxat:
	make -j -C LOBPCG/ run_sxat
	
clean:
	- make -j -C LOBPCG/ clean
