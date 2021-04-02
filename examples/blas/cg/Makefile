CXX=g++
CXXFLAGS=-O3 -fopenmp
INC=-I$(MONOLISH_DIR)/include/
LIB=-L$(MONOLISH_DIR)/lib/ 

all: cpu gpu

cpu: cg_cpu.out
gpu: cg_gpu.out
	
cg_cpu.out: cg.cpp
	$(CXX) $(INC) $(LIB) cg.cpp -o $@  -lmonolish_cpu

cg_gpu.out: cg.cpp
	$(CXX) $(INC) $(LIB) cg.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./cg_cpu.out
	echo "run GPU"
	./cg_gpu.out

clean:
	rm *.out
