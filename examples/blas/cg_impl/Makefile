CXX      ?= g++
CXXFLAGS  = -O3 -Wall

.PHONY: cpu gpu
all: cpu

cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

cg_gpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_gpu

cpu: cg_cpu.out
	./cg_cpu.out

gpu: cg_gpu.out
	./cg_gpu.out

clean:
	rm *.out
