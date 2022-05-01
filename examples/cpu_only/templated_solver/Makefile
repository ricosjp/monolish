CXX      ?= g++
# monlish requires C++11
CXXFLAGS ?= -O3 -Wall -std=c++11
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu
all: cpu

solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

cpu: solver_cpu.out
	./$<

clean:
	rm *.out
