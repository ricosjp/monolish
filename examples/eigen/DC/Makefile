MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: DC_cpu.out run_cpu
gpu: DC_gpu.out run_gpu
	
DC_cpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) DC.cpp -o $@ -lmonolish_cpu $(LIBS)

DC_gpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) DC.cpp -o $@ -lmonolish_gpu $(LIBS)

run_cpu:
	echo "run CPU"
	./DC_cpu.out

run_gpu:
	echo "run GPU"
	./DC_gpu.out

clean:
	rm *.out
