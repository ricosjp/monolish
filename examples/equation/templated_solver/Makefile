MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: solver_cpu.out run_cpu
gpu: solver_gpu.out run_gpu
	
solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@  -lmonolish_cpu $(LIBS)

solver_gpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@  -lmonolish_gpu $(LIBS)

run_cpu:
	echo "run CPU"
	./solver_cpu.out

run_gpu:
	echo "run GPU"
	./solver_gpu.out

clean:
	rm *.out
