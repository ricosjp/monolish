CXX=g++
CXXFLAGS=-O3

all: cpu gpu

cpu: DC_cpu.out
gpu: DC_gpu.out
	
DC_cpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) DC.cpp -o $@  -lmonolish_cpu

DC_gpu.out: DC.cpp
	$(CXX) $(CXXFLAGS) DC.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./DC_cpu.out
	echo "run GPU"
	./DC_gpu.out

clean:
	rm *.out
