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

VEC_TEST_SIZE = $(shell bash -c 'echo $$(($$RANDOM%10000+1000))')
VEC_TEST_ITER?=100
VEC_TEST_ANS_CHECK?=1

define RUN
	$(PROFILER) $(1) > `echo $(var) | sed 's/\.out//g'`.tsv
	cat  `echo $(var) | sed 's/\.out//g'`.tsv

endef

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
	make -B FUNC=$(<:.cpp=) ARCH=intel_oss -f ../test_cpu.mk

amd_oss: $(OBJS_AMD_OSS)
%_amd_oss.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=amd_oss -f ../test_cpu.mk

nvidia: $(OBJS_NVIDIA)
%_nvidia.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=nvidia -f ../test_gpu.mk

a64fx: $(OBJS_FX)
%_a64fx.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=a64fx -f ../test_a64fx.mk

sxat: $(OBJS_SX)
%_sxat.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=sxat -f ../test_sxat.mk

run_cpu:
	$(foreach var,$(OBJS_CPU), $(call RUN,$(var)))

run_gpu:
	$(foreach var,$(OBJS_GPU), $(call RUN,$(var)))

run_intel_mkl:
	$(foreach var,$(OBJS_INTEL_MKL), $(call RUN,$(var)))

run_intel_oss:
	$(foreach var,$(OBJS_INTEL_OSS), $(call RUN,$(var)))

run_amd_oss:
	$(foreach var,$(OBJS_AMD_OSS), $(call RUN,$(var)))
 
run_nvidia:
	$(foreach var,$(OBJS_NVIDIA), $(call RUN,$(var))) 

run_a64fx:
	$(foreach var,$(OBJS_FX), $(call RUN,$(var))) 

run_sxat:
	$(foreach var,$(OBJS_SX), $(call RUN,$(var))) 

clean:
	- rm *.out
	- rm *.tsv
