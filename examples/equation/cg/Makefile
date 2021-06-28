MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: cg_cpu.out run_cpu
gpu: cg_gpu.out run_gpu
	
cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) cg.cpp -o $@  -lmonolish_cpu $(LIBS)

cg_gpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) cg.cpp -o $@  -lmonolish_gpu $(LIBS)

run_cpu:
	echo "run CPU"
	./cg_cpu.out

run_gpu:
	echo "run GPU"
	./cg_gpu.out

clean:
	rm *.out
