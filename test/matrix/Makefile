SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_a64fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sxat.out))
.PHONY = cpu gpu a64fx sxat run_cpu run_gpu run_sx run_a64fx clean

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

a64fx: $(OBJS_FX)
%_a64fx.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=a64fx -f ../test_a64fx.mk

sxat: $(OBJS_SX)
%_sxat.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=sxat -f ../test_sxat.mk

run_cpu:
	bash ./test.sh matrix_common cpu
	bash ./test.sh matrix_convert cpu
	bash ./test.sh matrix_compare cpu
	bash ./test.sh matrix_transpose cpu
	bash ./test.sh matrix_blas cpu
	bash ./test.sh matrix_vml cpu
	bash ./test.sh matrix_subvec_op cpu
 
run_gpu:
	bash ./test.sh matrix_common gpu
	bash ./test.sh matrix_convert gpu
	bash ./test.sh matrix_compare gpu
	# bash ./test.sh matrix_transpose gpu
	bash ./test.sh matrix_blas gpu
	bash ./test.sh matrix_vml gpu
	bash ./test.sh matrix_subvec_op gpu

run_a64fx:
	bash ./test.sh matrix_common a64fx
	bash ./test.sh matrix_convert a64fx
	bash ./test.sh matrix_compare a64fx
	bash ./test.sh matrix_transpose a64fx
	bash ./test.sh matrix_blas a64fx
	bash ./test.sh matrix_vml a64fx
	bash ./test.sh matrix_subvec_op a64fx

run_sxat:
	# bash ./test.sh matrix_common sxat
	bash ./test.sh matrix_convert sxat
	bash ./test.sh matrix_compare sxat
	bash ./test.sh matrix_transpose sxat
	bash ./test.sh matrix_blas sxat
	bash ./test.sh matrix_vml sxat
	bash ./test.sh matrix_subvec_op sxat

clean:
	- rm *.out
