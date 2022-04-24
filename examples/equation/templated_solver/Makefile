CXX      ?= g++
CXXFLAGS ?= -O3 -Wall -std=c++17
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu gpu
all: cpu

solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

solver_gpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_gpu

cpu: solver_cpu.out
	./$<

gpu: solver_gpu.out
	./$<

clean:
	rm *.out
