FUNC=logging
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../Makefile_core.cpu

gpu:
	make -B FUNC=$(FUNC) ARCH=gpu -f ../../Makefile_core.gpu

run_cpu:
	./$(FUNC)_cpu.out monolish_test_cpu.log ../../test.mtx

run_gpu:
	$(PROFILER )./$(FUNC)_gpu.out monolish_test_gpu.log ../../test.mtx

log_cpu:
	# python3 logger.py monolish_test_cpu.log logging_result_cpu.html

log_gpu:
	# python3 logger.py monolish_test_gpu.log logging_result_gpu.html
	
clean:
	- rm *.out *.log *.html
