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
	bash ./benchmark.sh sm_add cpu
	bash ./benchmark.sh sm_sub cpu
	bash ./benchmark.sh sm_mul cpu
	bash ./benchmark.sh sm_div cpu
	bash ./benchmark.sh mm_add cpu
	bash ./benchmark.sh mm_sub cpu
	bash ./benchmark.sh mm_mul cpu
	bash ./benchmark.sh mm_div cpu
	bash ./benchmark.sh m_sqrt cpu
	bash ./benchmark.sh sm_pow cpu
	bash ./benchmark.sh mm_pow cpu
	bash ./benchmark.sh m_sin cpu
	bash ./benchmark.sh m_sinh cpu
	bash ./benchmark.sh m_asin cpu
	bash ./benchmark.sh m_asinh cpu
	bash ./benchmark.sh m_tan cpu
	bash ./benchmark.sh m_tanh cpu
	bash ./benchmark.sh m_atan cpu
	bash ./benchmark.sh m_atanh cpu
	bash ./benchmark.sh m_ceil cpu
	bash ./benchmark.sh m_floor cpu
	bash ./benchmark.sh m_sign cpu
 
run_gpu:
	bash ./benchmark.sh sm_add gpu
	bash ./benchmark.sh sm_sub gpu
	bash ./benchmark.sh sm_mul gpu
	bash ./benchmark.sh sm_div gpu
	bash ./benchmark.sh mm_add gpu
	bash ./benchmark.sh mm_sub gpu
	bash ./benchmark.sh mm_mul gpu
	bash ./benchmark.sh mm_div gpu
	bash ./benchmark.sh m_sin gpu
	bash ./benchmark.sh m_sinh gpu
	bash ./benchmark.sh m_asin gpu
	bash ./benchmark.sh m_asinh gpu
	bash ./benchmark.sh m_tan gpu
	bash ./benchmark.sh m_tanh gpu
	bash ./benchmark.sh m_atan gpu
	bash ./benchmark.sh m_atanh gpu
	bash ./benchmark.sh m_ceil gpu
	bash ./benchmark.sh m_floor gpu
	bash ./benchmark.sh m_sign gpu

run_fx:
	bash ./benchmark.sh sm_add fx
	bash ./benchmark.sh sm_sub fx
	bash ./benchmark.sh sm_mul fx
	bash ./benchmark.sh sm_div fx
	bash ./benchmark.sh mm_add fx
	bash ./benchmark.sh mm_sub fx
	bash ./benchmark.sh mm_mul fx
	bash ./benchmark.sh mm_div fx
	bash ./benchmark.sh m_sin fx
	bash ./benchmark.sh m_sinh fx
	bash ./benchmark.sh m_asin fx
	bash ./benchmark.sh m_asinh fx
	bash ./benchmark.sh m_tan fx
	bash ./benchmark.sh m_tanh fx
	bash ./benchmark.sh m_atan fx
	bash ./benchmark.sh m_atanh fx
	bash ./benchmark.sh m_ceil fx
	bash ./benchmark.sh m_floor fx
	bash ./benchmark.sh m_sign fx

run_sx:
	bash ./benchmark.sh sm_add sx
	bash ./benchmark.sh sm_sub sx
	bash ./benchmark.sh sm_mul sx
	bash ./benchmark.sh sm_div sx
	bash ./benchmark.sh mm_add sx
	bash ./benchmark.sh mm_sub sx
	bash ./benchmark.sh mm_mul sx
	bash ./benchmark.sh mm_div sx
	bash ./benchmark.sh m_sin sx
	bash ./benchmark.sh m_sinh sx
	bash ./benchmark.sh m_asin sx
	bash ./benchmark.sh m_asinh sx
	bash ./benchmark.sh m_tan sx
	bash ./benchmark.sh m_tanh sx
	bash ./benchmark.sh m_atan sx
	bash ./benchmark.sh m_atanh sx
	bash ./benchmark.sh m_ceil sx
	bash ./benchmark.sh m_floor sx
	bash ./benchmark.sh m_sign sx

clean:
	- rm *.out
	- rm *.tsv
