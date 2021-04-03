CXX=g++
CXXFLAGS=-O3 -fopenmp
INC=-I$(MONOLISH_DIR)/include/
LIB=-L$(MONOLISH_DIR)/lib/ 

all: cpu gpu

cpu: DC_cpu.out
gpu: DC_gpu.out
	
DC_cpu.out: DC.cpp
	$(CXX) $(INC) $(LIB) DC.cpp -o $@  -lmonolish_cpu

DC_gpu.out: DC.cpp
	$(CXX) $(INC) $(LIB) DC.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./DC_cpu.out
	echo "run GPU"
	./DC_gpu.out

clean:
	rm *.out
