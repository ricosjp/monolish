all: cpu

cpu:
	make -j -C Fortran/ cpu

gpu:
	make -j -C Fortran/ gpu

a64fx:
	make -j -C Fortran/ a64fx

sxat:
	make -j -C Fortran/ sxat

run_cpu:
	make -j -C Fortran/ run_cpu

run_gpu:
	make -j -C Fortran/ run_gpu
	
run_a64fx:
	make -j -C Fortran/ run_a64fx

run_sxat:
	make -j -C Fortran/ run_sxat
	
clean:
	- make -j -C Fortran/ clean

