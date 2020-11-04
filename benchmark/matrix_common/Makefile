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
	bash ./test.sh matrix_common cpu
	bash ./test.sh convert cpu
	bash ./test.sh mv_mul cpu
	bash ./test.sh transpose cpu
	bash ./test.sh subvec_op cpu
	bash ./test.sh m_tanh cpu
 
run_gpu:
	bash ./test.sh matrix_common gpu
	bash ./test.sh convert gpu
	bash ./test.sh mv_mul gpu
	bash ./test.sh transpose gpu
	bash ./test.sh m_tanh gpu
	#bash ./test.sh subvec_op gpu

run_fx:
	bash ./test.sh matrix_common fx
	bash ./test.sh convert fx
	bash ./test.sh mv_mul fx
	bash ./test.sh transpose fx
	bash ./test.sh subvec_op fx
	bash ./test.sh m_tanh fx

run_sx:
	bash ./test.sh matrix_common sx
	bash ./test.sh convert sx
	bash ./test.sh mv_mul sx
	bash ./test.sh transpose sx
	bash ./test.sh subvec_op sx
	bash ./test.sh m_tanh sx

clean:
	- rm *.out
