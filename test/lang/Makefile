all: cpu

cpu:
	make -j -C Fortran/ cpu

gpu:
	make -j -C Fortran/ gpu

fx:
	make -j -C Fortran/ fx

run_cpu:
	make -j -C Fortran/ run_cpu

run_gpu:
	make -j -C Fortran/ run_gpu
	
run_fx:
	make -j -C Fortran/ run_fx
	
clean:
	- make -j -C Fortran/ clean

