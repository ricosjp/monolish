SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_a64fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sxat.out))

OBJS_INTEL_MKL=$(addprefix ./, $(SRCS:.cpp=_intel_mkl.out))
OBJS_INTEL_OSS=$(addprefix ./, $(SRCS:.cpp=_intel_oss.out))
OBJS_AMD_OSS=$(addprefix ./, $(SRCS:.cpp=_amd_oss.out))
OBJS_NVIDIA=$(addprefix ./, $(SRCS:.cpp=_nvidia.out))

.PHONY = cpu gpu a64fx sxat run_cpu run_gpu run_sxat run_a64fx clean
.PHONY += intel_mkl intel_oss amd_oss nvidia
.PHONY += run_intel_mkl run_intel_oss run_amd_oss run_nvidia

all: cpu

cpu: $(OBJS_CPU)
%_cpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

gpu: $(OBJS_GPU)
%_gpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

intel_mkl: $(OBJS_INTEL_MKL)
%_intel_mkl.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=intel_mkl -f ../test_cpu.mk

intel_oss: $(OBJS_INTEL_OSS)
%_intel_oss.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

amd_oss: $(OBJS_AMD_OSS)
%_amd_oss.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

nvidia: $(OBJS_NVIDIA)
%_nvidia.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

a64fx: $(OBJS_FX)
%_a64fx.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=a64fx -f ../test_a64fx.mk

sxat: $(OBJS_SX)
%_sxat.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=sxat -f ../test_sxat.mk

run_cpu: 
	bash ./benchmark.sh sm_add intel_mkl
	bash ./benchmark.sh sm_sub intel_mkl
	bash ./benchmark.sh sm_mul intel_mkl
	bash ./benchmark.sh sm_div intel_mkl
	bash ./benchmark.sh mm_add intel_mkl
	bash ./benchmark.sh mm_sub intel_mkl
	bash ./benchmark.sh mm_mul intel_mkl
	bash ./benchmark.sh mm_div intel_mkl
	bash ./benchmark.sh m_sqrt intel_mkl
	bash ./benchmark.sh sm_pow intel_mkl
	bash ./benchmark.sh mm_pow intel_mkl
	bash ./benchmark.sh m_sin intel_mkl
	bash ./benchmark.sh m_sinh intel_mkl
	bash ./benchmark.sh m_asin intel_mkl
	bash ./benchmark.sh m_asinh intel_mkl
	bash ./benchmark.sh m_tan intel_mkl
	bash ./benchmark.sh m_tanh intel_mkl
	bash ./benchmark.sh m_atan intel_mkl
	bash ./benchmark.sh m_atanh intel_mkl
	bash ./benchmark.sh m_ceil intel_mkl
	bash ./benchmark.sh m_floor intel_mkl
	bash ./benchmark.sh m_sign intel_mkl
	bash ./benchmark.sh m_reciprocal intel_mkl

run_gpu:
	bash ./benchmark.sh sm_add nvidia
	bash ./benchmark.sh sm_sub nvidia
	bash ./benchmark.sh sm_mul nvidia
	bash ./benchmark.sh sm_div nvidia
	bash ./benchmark.sh mm_add nvidia
	bash ./benchmark.sh mm_sub nvidia
	bash ./benchmark.sh mm_mul nvidia
	bash ./benchmark.sh mm_div nvidia
	bash ./benchmark.sh m_sin nvidia
	bash ./benchmark.sh m_sinh nvidia
	bash ./benchmark.sh m_asin nvidia
	bash ./benchmark.sh m_asinh nvidia
	bash ./benchmark.sh m_tan nvidia
	bash ./benchmark.sh m_tanh nvidia
	bash ./benchmark.sh m_atan nvidia
	bash ./benchmark.sh m_atanh nvidia
	bash ./benchmark.sh m_ceil nvidia
	bash ./benchmark.sh m_floor nvidia
	bash ./benchmark.sh m_sign nvidia
	bash ./benchmark.sh m_reciprocal nvidia

run_intel_mkl: 
	bash ./benchmark.sh sm_add intel_mkl
	bash ./benchmark.sh sm_sub intel_mkl
	bash ./benchmark.sh sm_mul intel_mkl
	bash ./benchmark.sh sm_div intel_mkl
	bash ./benchmark.sh mm_add intel_mkl
	bash ./benchmark.sh mm_sub intel_mkl
	bash ./benchmark.sh mm_mul intel_mkl
	bash ./benchmark.sh mm_div intel_mkl
	bash ./benchmark.sh m_sqrt intel_mkl
	bash ./benchmark.sh sm_pow intel_mkl
	bash ./benchmark.sh mm_pow intel_mkl
	bash ./benchmark.sh m_sin intel_mkl
	bash ./benchmark.sh m_sinh intel_mkl
	bash ./benchmark.sh m_asin intel_mkl
	bash ./benchmark.sh m_asinh intel_mkl
	bash ./benchmark.sh m_tan intel_mkl
	bash ./benchmark.sh m_tanh intel_mkl
	bash ./benchmark.sh m_atan intel_mkl
	bash ./benchmark.sh m_atanh intel_mkl
	bash ./benchmark.sh m_ceil intel_mkl
	bash ./benchmark.sh m_floor intel_mkl
	bash ./benchmark.sh m_sign intel_mkl
	bash ./benchmark.sh m_reciprocal intel_mkl

run_intel_oss: 
	bash ./benchmark.sh sm_add intel_oss
	bash ./benchmark.sh sm_sub intel_oss
	bash ./benchmark.sh sm_mul intel_oss
	bash ./benchmark.sh sm_div intel_oss
	bash ./benchmark.sh mm_add intel_oss
	bash ./benchmark.sh mm_sub intel_oss
	bash ./benchmark.sh mm_mul intel_oss
	bash ./benchmark.sh mm_div intel_oss
	bash ./benchmark.sh m_sqrt intel_oss
	bash ./benchmark.sh sm_pow intel_oss
	bash ./benchmark.sh mm_pow intel_oss
	bash ./benchmark.sh m_sin intel_oss
	bash ./benchmark.sh m_sinh intel_oss
	bash ./benchmark.sh m_asin intel_oss
	bash ./benchmark.sh m_asinh intel_oss
	bash ./benchmark.sh m_tan intel_oss
	bash ./benchmark.sh m_tanh intel_oss
	bash ./benchmark.sh m_atan intel_oss
	bash ./benchmark.sh m_atanh intel_oss
	bash ./benchmark.sh m_ceil intel_oss
	bash ./benchmark.sh m_floor intel_oss
	bash ./benchmark.sh m_sign intel_oss
	bash ./benchmark.sh m_reciprocal intel_oss

run_amd_oss: 
	bash ./benchmark.sh sm_add amd_oss
	bash ./benchmark.sh sm_sub amd_oss
	bash ./benchmark.sh sm_mul amd_oss
	bash ./benchmark.sh sm_div amd_oss
	bash ./benchmark.sh mm_add amd_oss
	bash ./benchmark.sh mm_sub amd_oss
	bash ./benchmark.sh mm_mul amd_oss
	bash ./benchmark.sh mm_div amd_oss
	bash ./benchmark.sh m_sqrt amd_oss
	bash ./benchmark.sh sm_pow amd_oss
	bash ./benchmark.sh mm_pow amd_oss
	bash ./benchmark.sh m_sin amd_oss
	bash ./benchmark.sh m_sinh amd_oss
	bash ./benchmark.sh m_asin amd_oss
	bash ./benchmark.sh m_asinh amd_oss
	bash ./benchmark.sh m_tan amd_oss
	bash ./benchmark.sh m_tanh amd_oss
	bash ./benchmark.sh m_atan amd_oss
	bash ./benchmark.sh m_atanh amd_oss
	bash ./benchmark.sh m_ceil amd_oss
	bash ./benchmark.sh m_floor amd_oss
	bash ./benchmark.sh m_sign amd_oss
	bash ./benchmark.sh m_reciprocal amd_oss
 
nvidia:
	bash ./benchmark.sh sm_add nvidia
	bash ./benchmark.sh sm_sub nvidia
	bash ./benchmark.sh sm_mul nvidia
	bash ./benchmark.sh sm_div nvidia
	bash ./benchmark.sh mm_add nvidia
	bash ./benchmark.sh mm_sub nvidia
	bash ./benchmark.sh mm_mul nvidia
	bash ./benchmark.sh mm_div nvidia
	bash ./benchmark.sh m_sin nvidia
	bash ./benchmark.sh m_sinh nvidia
	bash ./benchmark.sh m_asin nvidia
	bash ./benchmark.sh m_asinh nvidia
	bash ./benchmark.sh m_tan nvidia
	bash ./benchmark.sh m_tanh nvidia
	bash ./benchmark.sh m_atan nvidia
	bash ./benchmark.sh m_atanh nvidia
	bash ./benchmark.sh m_ceil nvidia
	bash ./benchmark.sh m_floor nvidia
	bash ./benchmark.sh m_sign nvidia
	bash ./benchmark.sh m_reciprocal nvidia

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
