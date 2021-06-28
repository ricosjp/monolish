MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: cg_cpu.out run_cpu
gpu: cg_gpu.out run_gpu
	
cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu 

cg_gpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_gpu 

run_cpu:
	echo "run CPU"
	./cg_cpu.out

run_gpu:
	echo "run GPU"
	./cg_gpu.out

clean:
	rm *.out
