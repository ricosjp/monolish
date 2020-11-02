FUNC=cg
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../test_gpu.mk

fx:
	make -B FUNC=$(FUNC) ARCH=fx -f ../../test_fx.mk

sx:
	make -B FUNC=$(FUNC) ARCH=sx -f ../../test_sx.mk

run_cpu:
	./$(FUNC)_cpu.out ../../test.mtx 1

run_gpu:
	$(PROFILER)./$(FUNC)_gpu.out ../../test.mtx 1

run_fx:
	$(PROFILER)./$(FUNC)_fx.out ../../test.mtx 1
	
run_sx:
	$(PROFILER)./$(FUNC)_sx.out ../../test.mtx 1
	
clean:
	- rm *.out
