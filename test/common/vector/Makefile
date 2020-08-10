FUNC=vector_common
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

fx:
	make -B FUNC=$(FUNC) ARCH=fx -f ../../Makefile_core.fx

run_cpu:
	./$(FUNC)_cpu.out 10

run_gpu:
	$(PROFILER) ./$(FUNC)_gpu.out 10

run_fx:
	./$(FUNC)_fx.out 10
	
clean:
	- rm *.out
