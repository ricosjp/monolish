SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sx.out))
.PHONY = cpu gpu fx sx run_cpu run_gpu run_sx run_fx clean

all: cpu

cpu: $(OBJS_CPU)
%_cpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

gpu: $(OBJS_GPU)
%_gpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

fx: $(OBJS_FX)
%_fx.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=fx -f ../test_fx.mk

sx: $(OBJS_SX)
%_sx.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=sx -f ../test_sx.mk

run_cpu:
	bash ./benchmark.sh matadd cpu
	bash ./benchmark.sh mscal cpu
	bash ./benchmark.sh matvec cpu
	bash ./benchmark.sh matmul cpu
 
run_gpu:
	bash ./benchmark.sh matadd gpu
	bash ./benchmark.sh mscal gpu
	bash ./benchmark.sh matvec gpu
	bash ./benchmark.sh matmul gpu

run_fx:
	bash ./benchmark.sh matadd fx
	bash ./benchmark.sh mscal fx
	bash ./benchmark.sh matvec fx
	bash ./benchmark.sh matmul fx

run_sx:
	bash ./benchmark.sh matadd sx
	bash ./benchmark.sh mscal sx
	bash ./benchmark.sh matvec sx
	bash ./benchmark.sh matmul sx

clean:
	- rm *.out
	- rm *.tsv
