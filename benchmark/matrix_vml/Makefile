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
	bash ./benchmark.sh m_reciprocal cpu
 
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
	bash ./benchmark.sh m_reciprocal gpu

run_a64fx:
	bash ./benchmark.sh sm_add a64fx
	bash ./benchmark.sh sm_sub a64fx
	bash ./benchmark.sh sm_mul a64fx
	bash ./benchmark.sh sm_div a64fx
	bash ./benchmark.sh mm_add a64fx
	bash ./benchmark.sh mm_sub a64fx
	bash ./benchmark.sh mm_mul a64fx
	bash ./benchmark.sh mm_div a64fx
	bash ./benchmark.sh m_sin a64fx
	bash ./benchmark.sh m_sinh a64fx
	bash ./benchmark.sh m_asin a64fx
	bash ./benchmark.sh m_asinh a64fx
	bash ./benchmark.sh m_tan a64fx
	bash ./benchmark.sh m_tanh a64fx
	bash ./benchmark.sh m_atan a64fx
	bash ./benchmark.sh m_atanh a64fx
	bash ./benchmark.sh m_ceil a64fx
	bash ./benchmark.sh m_floor a64fx
	bash ./benchmark.sh m_sign a64fx
	bash ./benchmark.sh m_reciprocal a64fx

run_sxat:
	bash ./benchmark.sh sm_add sxat
	bash ./benchmark.sh sm_sub sxat
	bash ./benchmark.sh sm_mul sxat
	bash ./benchmark.sh sm_div sxat
	bash ./benchmark.sh mm_add sxat
	bash ./benchmark.sh mm_sub sxat
	bash ./benchmark.sh mm_mul sxat
	bash ./benchmark.sh mm_div sxat
	bash ./benchmark.sh m_sin sxat
	bash ./benchmark.sh m_sinh sxat
	bash ./benchmark.sh m_asin sxat
	bash ./benchmark.sh m_asinh sxat
	bash ./benchmark.sh m_tan sxat
	bash ./benchmark.sh m_tanh sxat
	bash ./benchmark.sh m_atan sxat
	bash ./benchmark.sh m_atanh sxat
	bash ./benchmark.sh m_ceil sxat
	bash ./benchmark.sh m_floor sxat
	bash ./benchmark.sh m_sign sxat
	bash ./benchmark.sh m_reciprocal sxat

clean:
	- rm *.out
	- rm *.tsv
