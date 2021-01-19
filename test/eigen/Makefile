all: cpu

cpu:
	make -j -C LOBPCG/ cpu
	make -j -C DC/ cpu

gpu:
	make -j -C LOBPCG/ gpu
	make -j -C DC/ gpu

a64fx:
	make -j -C LOBPCG/ a64fx
	make -j -C DC/ a64fx

sxat:
	make -j -C LOBPCG/ sxat
	make -j -C DC/ sxat

run_cpu:
	make -j -C LOBPCG/ run_cpu
	make -j -C DC/ run_cpu

run_gpu:
	make -j -C LOBPCG/ run_gpu
	make -j -C DC/ run_gpu

run_a64fx:
	make -j -C LOBPCG/ run_a64fx
	make -j -C DC/ run_a64fx

run_sxat:
	make -j -C LOBPCG/ run_sxat
	make -j -C DC/ run_sxat
	
clean:
	- make -j -C LOBPCG/ clean
	- make -j -C DC/ clean
