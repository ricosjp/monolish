CXX      ?= g++
CXXFLAGS ?= -O3 -Wall -std=c++17
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu gpu
all: cpu

lobpcg_cpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

lobpcg_gpu.out: lobpcg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_gpu

cpu: lobpcg_cpu.out
	./$<

gpu: lobpcg_gpu.out
	./$<

clean:
	rm *.out
