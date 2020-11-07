FUNC=logging
MONOLISH_LOG_VIEWER ?= $(join $(subst /test/logger/logging,,$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))),/python)

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
	./$(FUNC)_cpu.out monolish_test_cpu.log ../../test.mtx

run_gpu:
	$(PROFILER )./$(FUNC)_gpu.out monolish_test_gpu.log ../../test.mtx

run_fx:
	$(PROFILER )./$(FUNC)_fx.out monolish_test_fx.log ../../test.mtx

clean:
	- rm *.out *.log *.html
