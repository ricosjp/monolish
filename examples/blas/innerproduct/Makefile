CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu gpu
all: cpu

innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

innerproduct_gpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_gpu

cpu: innerproduct_cpu.out
	./$<

gpu: innerproduct_gpu.out
	./$<

clean:
	rm *.out
