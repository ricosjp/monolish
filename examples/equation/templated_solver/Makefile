CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: solver_cpu.out
gpu: solver_gpu.out
	
solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@  -lmonolish_cpu

solver_gpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) solver.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./solver_cpu.out
	echo "run GPU"
	./solver_gpu.out

clean:
	rm *.out
