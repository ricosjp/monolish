SRCS =$(notdir $(wildcard  ./*.cpp))
vpath %.cpp ./

OBJS_CPU=$(addprefix ./, $(SRCS:.cpp=_cpu.out))
OBJS_GPU=$(addprefix ./, $(SRCS:.cpp=_gpu.out))
OBJS_FX=$(addprefix ./, $(SRCS:.cpp=_a64fx.out))
OBJS_SX=$(addprefix ./, $(SRCS:.cpp=_sxat.out))
.PHONY = cpu gpu a64fx sxat run_cpu run_gpu run_sxat run_a64fx clean

all: cpu

intel_mkl: $(OBJS_CPU)
%_cpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

intel_oss: $(OBJS_CPU)
%_cpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

amd_oss: $(OBJS_CPU)
%_cpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=cpu -f ../test_cpu.mk

nvidia: $(OBJS_GPU)
%_gpu.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=gpu -f ../test_gpu.mk

a64fx: $(OBJS_FX)
%_a64fx.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=a64fx -f ../test_a64fx.mk

sxat: $(OBJS_SX)
%_sxat.out: %.cpp
	make -B FUNC=$(<:.cpp=) ARCH=sxat -f ../test_sxat.mk

run_intel_mkl:
	bash ./benchmark.sh matcopy intel_mkl
	bash ./benchmark.sh matadd intel_mkl
	bash ./benchmark.sh matsub intel_mkl
	bash ./benchmark.sh mscal intel_mkl
	bash ./benchmark.sh matvec intel_mkl
	bash ./benchmark.sh matmul intel_mkl

run_intel_oss:
	bash ./benchmark.sh matcopy intel_oss
	bash ./benchmark.sh matadd intel_oss
	bash ./benchmark.sh matsub intel_oss
	bash ./benchmark.sh mscal intel_oss
	bash ./benchmark.sh matvec intel_oss
	bash ./benchmark.sh matmul intel_oss

run_amd_oss:
	bash ./benchmark.sh matcopy amd_oss
	bash ./benchmark.sh matadd amd_oss
	bash ./benchmark.sh matsub amd_oss
	bash ./benchmark.sh mscal amd_oss
	bash ./benchmark.sh matvec amd_oss
	bash ./benchmark.sh matmul amd_oss
 
run_nvidia:
	bash ./benchmark.sh matcopy nvidia
	bash ./benchmark.sh matadd nvidia
	bash ./benchmark.sh matsub nvidia
	bash ./benchmark.sh mscal nvidia
	bash ./benchmark.sh matvec nvidia
	bash ./benchmark.sh matmul nvidia

run_a64fx:
	bash ./benchmark.sh matcopy a64fx
	bash ./benchmark.sh matadd a64fx
	bash ./benchmark.sh matsub a64fx
	bash ./benchmark.sh mscal a64fx
	bash ./benchmark.sh matvec a64fx
	bash ./benchmark.sh matmul a64fx

run_sxat:
	bash ./benchmark.sh matcopy sxat
	bash ./benchmark.sh matadd sxat
	bash ./benchmark.sh matsub sxat
	bash ./benchmark.sh mscal sxat
	bash ./benchmark.sh matvec sxat
	bash ./benchmark.sh matmul sxat

clean:
	- rm *.out
	- rm *.tsv
