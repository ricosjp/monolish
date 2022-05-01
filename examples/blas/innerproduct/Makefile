CXX      ?= g++
# monlish requires C++11
CXXFLAGS ?= -O3 -Wall -std=c++11
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu gpu
all: cpu

innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

innerproduct_gpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_gpu

cpu: innerproduct_cpu.out
	./$<

gpu: innerproduct_gpu.out
	./$<

clean:
	rm *.out
