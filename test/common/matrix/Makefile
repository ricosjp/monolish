FUNC=matrix_common
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

run_cpu:
	./$(FUNC)_cpu.out 10

run_gpu:
	$(PROFILER) ./$(FUNC)_gpu.out 10
	
clean:
	- rm *.out
