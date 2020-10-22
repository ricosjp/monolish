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
	bash ./test.sh matrix_common cpu 10
	bash ./test.sh matrix_convert cpu 10
	bash ./test.sh matrix_transpose cpu 10
	bash ./test.sh matrix_blas cpu 10
	bash ./test.sh matrix_arithmetic cpu 10
	bash ./test.sh matrix_subvec_op cpu 10
	bash ./test.sh matrix_math cpu 10
 
run_gpu:
	bash ./test.sh matrix_common gpu 10
	bash ./test.sh matrix_convert gpu 10
	bash ./test.sh matrix_transpose gpu 10
	bash ./test.sh matrix_blas gpu 10
	bash ./test.sh matrix_arithmetic gpu 10
	bash ./test.sh matrix_subvec_op gpu 10
	bash ./test.sh matrix_math gpu 10

run_fx:
	bash ./test.sh matrix_common fx 10
	bash ./test.sh matrix_convert fx 10
	bash ./test.sh matrix_transpose fx 10
	bash ./test.sh matrix_blas fx 10
	bash ./test.sh matrix_arithmetic fx 10
	bash ./test.sh matrix_subvec_op fx 10
	bash ./test.sh matrix_math fx 10

run_sx:
	bash ./test.sh matrix_common sx 10
	bash ./test.sh matrix_convert sx 10
	bash ./test.sh matrix_transpose sx 10
	bash ./test.sh matrix_blas sx 10
	bash ./test.sh matrix_arithmetic sx 10
	bash ./test.sh matrix_subvec_op sx 10
	bash ./test.sh matrix_math sx 10

clean:
	- rm *.out
