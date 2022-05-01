CXX      ?= g++
# monlish requires C++11
CXXFLAGS ?= -O3 -Wall -std=c++11
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu
all: cpu

cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

cpu: cg_cpu.out
	./$<

clean:
	rm *.out
