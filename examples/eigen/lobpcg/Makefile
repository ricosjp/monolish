MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: lobpcg_cpu.out run_cpu
gpu: lobpcg_gpu.out run_gpu
	
lobpcg_cpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) lobpcg.cpp -o $@ -lmonolish_cpu $(LIBS)

lobpcg_gpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) lobpcg.cpp -o $@ -lmonolish_gpu $(LIBS)

run_cpu:
	echo "run CPU"
	./lobpcg_cpu.out

run_gpu:
	echo "run GPU"
	./lobpcg_gpu.out

clean:
	rm *.out
