FUNC=logging
MONOLISH_LOG_VIEWER ?= $(join $(subst /test/logger/logging,,$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))),/python)

all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../test_gpu.mk

a64fx:
	make -B FUNC=$(FUNC) ARCH=a64fx -f ../../test_a64fx.mk

sxat:
	make -B FUNC=$(FUNC) ARCH=sxat -f ../../test_sxat.mk

run_cpu:
	./$(FUNC)_cpu.out monolish_test_cpu.log ../../test.mtx

run_gpu:
	$(PROFILER )./$(FUNC)_gpu.out monolish_test_gpu.log ../../test.mtx

run_a64fx:
	$(PROFILER )./$(FUNC)_a64fx.out monolish_test_a64fx.log ../../test.mtx

clean:
	- rm *.out *.log *.html
