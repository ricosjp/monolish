CXX      ?= g++
CXXFLAGS ?= -O3 -Wall

.PHONY: cpu gpu
all: cpu

DC_cpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_cpu

DC_gpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lmonolish_gpu

cpu: DC_cpu.out
	./$<

gpu: DC_gpu.out
	./$<

clean:
	rm *.out
