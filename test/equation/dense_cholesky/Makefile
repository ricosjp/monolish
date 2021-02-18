FUNC=chol
all:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

cpu:
	make -B FUNC=$(FUNC) ARCH=cpu -f ../../test_cpu.mk

#gpu:
#	make -B FUNC=$(FUNC) ARCH=gpu -f ../../test_gpu.mk

a64fx:
	make -B FUNC=$(FUNC) ARCH=a64fx -f ../../test_a64fx.mk

sxat:
	make -B FUNC=$(FUNC) ARCH=sxat -f ../../test_sxat.mk

run_cpu:
	./$(FUNC)_cpu.out ../../test.mtx 1

#run_gpu:
#	$(PROFILER)./$(FUNC)_gpu.out ../../test.mtx 1

run_a64fx:
	$(PROFILER)./$(FUNC)_a64fx.out ../../test.mtx 1
	
run_sxat:
	$(PROFILER)./$(FUNC)_sxat.out ../../test.mtx 1
	
clean:
	- rm *.out
