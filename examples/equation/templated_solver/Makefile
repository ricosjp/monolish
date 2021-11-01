CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu gpu
all: cpu

solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

solver_gpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_gpu

cpu: solver_cpu.out
	./$<

gpu: solver_gpu.out
	./$<

clean:
	rm *.out
