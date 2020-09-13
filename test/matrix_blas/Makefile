SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sx.out))
.PHONY = cpu gpu fx sx run_cpu run_gpu run_sx run_fx clean

MAT_TEST_SIZE?=123
MAT_TEST_SIZE?=124
MAT_TEST_ITER?=100
MAT_TEST_ANS_CHECK?=1

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
	bash ./test.sh matmul cpu 10
	bash ./test.sh matvec cpu 10
	bash ./test.sh mscal cpu 10
 
run_gpu:
	bash ./test.sh matmul gpu 10
	bash ./test.sh matvec gpu 10
	bash ./test.sh mscal gpu 10

run_fx:
	bash ./test.sh matmul fx 10
	bash ./test.sh matvec fx 10
	bash ./test.sh mscal fx 10

run_sx:
	bash ./test.sh matmul sx 10
	bash ./test.sh matvec sx 10
	bash ./test.sh mscal sx 10

clean:
	- rm *.out
