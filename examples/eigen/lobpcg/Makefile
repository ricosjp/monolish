MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: lobpcg_cpu.out run_cpu
gpu: lobpcg_gpu.out run_gpu
	
lobpcg_cpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu

lobpcg_gpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_gpu 

run_cpu:
	echo "run CPU"
	./lobpcg_cpu.out

run_gpu:
	echo "run GPU"
	./lobpcg_gpu.out

clean:
	rm *.out
