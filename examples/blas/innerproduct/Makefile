CXX=g++
CXXFLAGS=-O3 -fopenmp
INC=-I$(MONOLISH_DIR)/include/
LIB=-L$(MONOLISH_DIR)/lib/ 

all: cpu gpu

cpu: innerproduct_cpu.out
gpu: innerproduct_gpu.out
	
innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(INC) $(LIB) innerproduct.cpp -o $@  -lmonolish_cpu

innerproduct_gpu.out: innerproduct.cpp
	$(CXX) $(INC) $(LIB) innerproduct.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./innerproduct_cpu.out
	echo "run GPU"
	./innerproduct_gpu.out

clean:
	rm *.out
