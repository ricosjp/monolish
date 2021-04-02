CXX=g++
CXXFLAGS=-O3 -fopenmp
INC=-I$(MONOLISH_DIR)/include/
LIB=-L$(MONOLISH_DIR)/lib/ 

all: cpu

cpu: cg_cpu.out
	
cg_cpu.out: cg.cpp
	$(CXX) $(INC) $(LIB) cg.cpp -o $@  -lmonolish_gpu

