CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu
all: cpu

innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

cpu: innerproduct_cpu.out
	./$<

clean:
	rm *.out
