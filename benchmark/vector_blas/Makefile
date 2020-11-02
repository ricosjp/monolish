SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sx.out))

FILENAME=`echo $(1) | sed 's/\.out//g'` 
.PHONY = cpu gpu fx sx run_cpu run_gpu run_sx run_fx clean

define RUN
	$(PROFILER) $(1)  | tee `echo $(1) | sed 's/\.out//g'`.tsv

endef

all: cpu

cpu: $(OBJS_CPU)
%_cpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

gpu: $(OBJS_GPU)
%_gpu.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

fx: $(OBJS_FX)
%_fx.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=fx -f ../test_fx.mk

sx: $(OBJS_SX)
%_sx.out: %.cpp ../benchmark_utils.hpp
	make -B FUNC=$(<:.cpp=) ARCH=sx -f ../test_sx.mk

run_cpu:
	$(foreach var,$(OBJS_CPU), $(call RUN,$(var)))
 
run_gpu:
	$(foreach var,$(OBJS_GPU), $(call RUN,$(var)))

run_fx:
	$(foreach var,$(OBJS_FX), $(call RUN,$(var)))

run_sx:
	$(foreach var,$(OBJS_SX), $(call RUN,$(var)))

clean:
	- rm *.out
	- rm *.tsv
