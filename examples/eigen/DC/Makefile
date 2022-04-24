CXX      ?= g++
CXXFLAGS ?= -O3 -Wall -std=c++17
LDFLAGS ?=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ 

.PHONY: cpu gpu
all: cpu

DC_cpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_cpu

DC_gpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lmonolish_gpu

cpu: DC_cpu.out
	./$<

gpu: DC_gpu.out
	./$<

clean:
	rm *.out
