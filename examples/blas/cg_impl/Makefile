CXX      ?= g++
# monlish requires C++11
CXXFLAGS ?= -O3 -Wall -std=c++11
CXXFLAGS ?= -O3 -Wall -std=c++11
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu gpu
all: cpu

cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

cg_gpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_gpu

cpu: cg_cpu.out
	./$<

gpu: cg_gpu.out
	./$<

clean:
	rm *.out
