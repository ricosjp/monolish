FUNC=logging
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

fx:
	make -B FUNC=$(FUNC) ARCH=fx -f ../../Makefile_core.fx

run_cpu:
	./$(FUNC)_cpu.out monolish_test_cpu.log ../../test.mtx

run_gpu:
	$(PROFILER )./$(FUNC)_gpu.out monolish_test_gpu.log ../../test.mtx

run_fx:
	$(PROFILER )./$(FUNC)_fx.out monolish_test_fx.log ../../test.mtx

log_cpu:
	python3 logger.py monolish_test_cpu.log logging_result_cpu.html

log_gpu:
	python3 logger.py monolish_test_gpu.log logging_result_gpu.html

log_fx:
	python3 logger.py monolish_test_fx.log logging_result_fx.html
	
clean:
	- rm *.out *.log *.html
