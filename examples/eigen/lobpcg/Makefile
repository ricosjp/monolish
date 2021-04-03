CXX=g++
CXXFLAGS=-O3 -fopenmp
INC=-I$(MONOLISH_DIR)/include/
LIB=-L$(MONOLISH_DIR)/lib/ 

all: cpu gpu

cpu: lobpcg_cpu.out
gpu: lobpcg_gpu.out
	
lobpcg_cpu.out: lobpcg.cpp
	$(CXX) $(INC) $(LIB) lobpcg.cpp -o $@  -lmonolish_cpu

lobpcg_gpu.out: lobpcg.cpp
	$(CXX) $(INC) $(LIB) lobpcg.cpp -o $@  -lmonolish_gpu

run:
	echo "run CPU"
	./lobpcg_cpu.out
	echo "run GPU"
	./lobpcg_gpu.out

clean:
	rm *.out
