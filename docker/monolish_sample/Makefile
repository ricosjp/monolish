CXX=g++
CXXFLAGS=-O3 -lm -std=c++11
GPUFLAGS+=-DUSE_GPU -fopenacc -foffload=nvptx-none 

all:cpu

cpu:
	$(CXX) $(CXXFLAGS) dot.cpp -o dot_$@.out -lmonolish_$@

gpu:
	$(CXX) $(CXXFLAGS) $(GPUFLAGS) dot.cpp -o dot_$@.out -lmonolish_$@
	$(CXX) $(CXXFLAGS) $(GPUFLAGS) slu.cpp -o slu_$@.out -lmonolish_$@

clean:
	- rm *.out
