CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu
all: cpu

solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

cpu: solver_cpu.out
	./$<

clean:
	rm *.out
