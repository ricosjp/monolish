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

define RUN_matmul
	./$(1) $(2) $(3) Dense Dense $(MAT_TEST_SIZE) $(MAT_TEST_SIZE) $(MAT_TEST_SIZE) $(MAT_TEST_ITER) $(MAT_TEST_ANS_CHECK)

endef

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
	bash ./test.sh matrix_common cpu
	bash ./test.sh matrix_convert cpu
	bash ./test.sh matrix_transpose cpu
	bash ./test.sh matrix_blas cpu
	bash ./test.sh matrix_vml cpu
	bash ./test.sh matrix_subvec_op cpu
 
run_gpu:
	bash ./test.sh matrix_common gpu
	bash ./test.sh matrix_convert gpu
	bash ./test.sh matrix_transpose gpu
	bash ./test.sh matrix_blas gpu
	bash ./test.sh matrix_vml gpu
	bash ./test.sh matrix_subvec_op gpu

run_fx:
	bash ./test.sh matrix_common fx
	bash ./test.sh matrix_convert fx
	bash ./test.sh matrix_transpose fx
	bash ./test.sh matrix_blas fx
	bash ./test.sh matrix_vml fx
	bash ./test.sh matrix_subvec_op fx

run_sx:
	bash ./test.sh matrix_common sx
	bash ./test.sh matrix_convert sx
	bash ./test.sh matrix_transpose sx
	bash ./test.sh matrix_blas sx
	bash ./test.sh matrix_vml sx
	bash ./test.sh matrix_subvec_op sx

clean:
	- rm *.out
