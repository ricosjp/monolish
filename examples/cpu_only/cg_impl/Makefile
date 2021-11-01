CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu
all: cpu

cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

cpu: cg_cpu.out
	./$<

clean:
	rm *.out
