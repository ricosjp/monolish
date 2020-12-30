SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_a64fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sxat.out))

.PHONY = cpu gpu a64fx sxat run_cpu run_gpu run_sxat run_a64fx clean

define RUN
	$(PROFILER) $(1) > `echo $(var) | sed 's/\.out//g'`.tsv
	cat `echo $(var) | sed 's/\.out//g'`.tsv

endef

all: cpu

cpu: $(OBJS_CPU)
%_cpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

gpu: $(OBJS_GPU)
%_gpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

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

run_a64fx:
	$(foreach var,$(OBJS_FX), $(call RUN,$(var))) 

run_sxat:
	$(foreach var,$(OBJS_SX), $(call RUN,$(var))) 

clean:
	- rm *.out
	- rm *.tsv
