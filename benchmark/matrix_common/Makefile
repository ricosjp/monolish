SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_a64fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sxat.out))
.PHONY = cpu gpu a64fx sxat run_cpu run_gpu run_sxat run_a64fx clean

all: cpu

cpu: $(OBJS_CPU)
%_cpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

gpu: $(OBJS_GPU)
%_gpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

a64fx: $(OBJS_FX)
%_a64fx.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=a64fx -f ../test_a64fx.mk

sxat: $(OBJS_SX)
%_sxat.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=sxat -f ../test_sxat.mk

run_cpu:
	bash ./benchmark.sh convert cpu
	bash ./benchmark.sh transpose cpu
 
run_gpu:
	#bash ./benchmark.sh convert gpu
	#bash ./benchmark.sh transpose gpu

run_a64fx:
	bash ./benchmark.sh convert a64fx
	bash ./benchmark.sh transpose a64fx

run_sxat:
	bash ./benchmark.sh convert sxat
	bash ./benchmark.sh transpose sxat

clean:
	- rm *.out
	- rm *.tsv
