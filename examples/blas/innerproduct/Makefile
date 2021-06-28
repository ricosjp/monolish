MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: innerproduct_cpu.out run_cpu
gpu: innerproduct_gpu.out run_gpu
	
innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) innerproduct.cpp -o $@  -lmonolish_cpu $(LIBS)

innerproduct_gpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) innerproduct.cpp -o $@  -lmonolish_gpu $(LIBS)

run_cpu:
	echo "run CPU"
	./innerproduct_cpu.out

run_gpu:
	echo "run GPU"
	./innerproduct_gpu.out

clean:
	rm *.out
