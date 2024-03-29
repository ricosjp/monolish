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

run_cpu: cpu
	bash ./benchmark.sh LU cpu Dense
	bash ./benchmark.sh CG cpu CRS
	bash ./benchmark.sh BiCGSTAB cpu CRS
	bash ./benchmark.sh Jacobi cpu CRS
	bash ./benchmark.sh SOR cpu CRS

run_gpu: 
	bash ./benchmark.sh LU gpu Dense
	bash ./benchmark.sh CG gpu CRS
	bash ./benchmark.sh BiCGSTAB gpu CRS
	bash ./benchmark.sh Jacobi gpu CRS
	bash ./benchmark.sh SOR gpu CRS

run_intel_mkl: intel_mkl
	bash ./benchmark.sh LU intel_mkl Dense
	bash ./benchmark.sh CG intel_mkl CRS
	bash ./benchmark.sh BiCGSTAB intel_mkl CRS
	bash ./benchmark.sh Jacobi intel_mkl CRS
	bash ./benchmark.sh SOR intel_mkl CRS

run_intel_oss: intel_oss
	bash ./benchmark.sh LU intel_oss Dense
	bash ./benchmark.sh CG intel_oss CRS
	bash ./benchmark.sh BiCGSTAB intel_oss CRS
	bash ./benchmark.sh Jacobi intel_oss CRS
	bash ./benchmark.sh SOR intel_oss CRS

run_amd_oss: amd_oss
	bash ./benchmark.sh LU amd_oss Dense
	bash ./benchmark.sh CG amd_oss CRS
	bash ./benchmark.sh BiCGSTAB amd_oss CRS
	bash ./benchmark.sh Jacobi amd_oss CRS
	bash ./benchmark.sh SOR amd_oss CRS

run_nvidia: nvidia
	bash ./benchmark.sh LU nvidia Dense
	bash ./benchmark.sh CG nvidia CRS
	bash ./benchmark.sh BiCGSTAB nvidia CRS
	bash ./benchmark.sh Jacobi nvidia CRS
	bash ./benchmark.sh SOR nvidia CRS

run_a64fx: a64fx
	bash ./benchmark.sh LU a64fx Dense
	bash ./benchmark.sh CG a64fx CRS
	bash ./benchmark.sh BiCGSTAB a64fx CRS
	bash ./benchmark.sh Jacobi a64fx CRS
	bash ./benchmark.sh SOR a64fx CRS

run_sxat: sxat
	bash ./benchmark.sh LU sxat Dense
	bash ./benchmark.sh CG sxat CRS
	bash ./benchmark.sh BiCGSTAB sxat CRS
	bash ./benchmark.sh Jacobi sxat CRS
	bash ./benchmark.sh SOR sxat CRS

clean:
	- rm *.out
	- rm *.tsv
