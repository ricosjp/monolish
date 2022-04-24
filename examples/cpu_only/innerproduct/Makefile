CXX      ?= g++
CXXFLAGS ?= -O3 -Wall -std=c++17
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu
all: cpu

innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

cpu: innerproduct_cpu.out
	./$<

clean:
	rm *.out
