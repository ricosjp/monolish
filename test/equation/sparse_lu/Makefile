FUNC=slu
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

fx:
	make -B FUNC=$(FUNC) ARCH=fx -f ../../Makefile_core.fx

run_cpu:
	./$(FUNC)_cpu.out ../../test.mtx 1

run_gpu:
	$(PROFILER) ./$(FUNC)_gpu.out ../../test.mtx 1

run_fx:
	$(PROFILER) ./$(FUNC)_fx.out ../../test.mtx 1
	
clean:
	- rm *.out
