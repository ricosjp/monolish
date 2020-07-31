FUNC=matmul_dense
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

run_cpu:
	./$(FUNC)_cpu.out 100 100 100 10 1
	./$(FUNC)_cpu.out 101 100 100 10 1
	./$(FUNC)_cpu.out 100 101 100 10 1
	./$(FUNC)_cpu.out 100 100 101 10 1

run_gpu:
	$(PROFILER) ./$(FUNC)_gpu.out 100 100 100 10 1
	$(PROFILER) ./$(FUNC)_gpu.out 101 100 100 10 1
	$(PROFILER) ./$(FUNC)_gpu.out 100 101 100 10 1
	$(PROFILER) ./$(FUNC)_gpu.out 100 100 101 10 1
	
clean:
	- rm *.out
