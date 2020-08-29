FUNC=dot
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

fx:
	make -B FUNC=$(FUNC) ARCH=fx -f ../../Makefile_core.fx

sx:
	make -B FUNC=$(FUNC) ARCH=sx -f ../../Makefile_core.sx

run_cpu:
	./$(FUNC)_cpu.out 10000 10 1

run_gpu:
	$(PROFILER) ./$(FUNC)_gpu.out 10000 10 1
	
run_fx:
	$(PROFILER) ./$(FUNC)_fx.out 10000 10 1
	
run_sx:
	$(PROFILER) ./$(FUNC)_sx.out 10000 10 1
	
clean:
	- rm *.out
