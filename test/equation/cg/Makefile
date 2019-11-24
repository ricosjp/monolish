FUNC=cg
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

run_cpu:
	./$(FUNC)_cpu.out ../../test.mtx

run_gpu:
	./$(FUNC)_gpu.out ../../test.mtx
	
clean:
	- rm *.out
