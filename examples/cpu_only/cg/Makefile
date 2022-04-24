CXX      ?= g++
CXXFLAGS ?= -O3 -Wall -std=c++17
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu
all: cpu

cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

cpu: cg_cpu.out
	./$<

clean:
	rm *.out
