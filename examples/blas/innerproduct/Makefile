MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: innerproduct_cpu.out run_cpu
gpu: innerproduct_gpu.out run_gpu
	
innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu 

innerproduct_gpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_gpu 

run_cpu:
	echo "run CPU"
	./innerproduct_cpu.out

run_gpu:
	echo "run GPU"
	./innerproduct_gpu.out

clean:
	rm *.out
