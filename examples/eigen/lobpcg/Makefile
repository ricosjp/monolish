CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu gpu
all: cpu

lobpcg_cpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

lobpcg_gpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_gpu

cpu: lobpcg_cpu.out
	./$<

gpu: lobpcg_gpu.out
	./$<

clean:
	rm *.out
