SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sx.out))
.PHONY = cpu gpu fx sx run_cpu run_gpu run_sx run_fx clean

VEC_TEST_SIZE?=12345
VEC_TEST_ITER?=100
VEC_TEST_ANS_CHECK?=1

define RUN
	$(1) double $(VEC_TEST_SIZE) $(VEC_TEST_ITER) $(VEC_TEST_ANS_CHECK)
	$(1) float $(VEC_TEST_SIZE) $(VEC_TEST_ITER) $(VEC_TEST_ANS_CHECK)

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
	$(foreach var,$(OBJS_CPU), $(call RUN,$(var)))
 
run_gpu:
	$(foreach var,$(OBJS_GPU), $(call RUN,$(var)))

run_fx:
	$(foreach var,$(OBJS_FX), $(call RUN,$(var)))

run_sx:
	$(foreach var,$(OBJS_SX), $(call RUN,$(var)))

clean:
	- rm *.out
