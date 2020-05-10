MONOLISH_DIR?=$(HOME)/lib/monolish

FCC ?= gfortran
FFLAGS=-O3 -fopenmp

CXX ?= g++
CXXFLAGS=-O3 -lm -std=c++17 -fopenmp

LIB_CPU=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu
LIB_GPU=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu

LDFLAGS = -lstdc++ -lmonolish

cpu:
	$(FCC) $(FFLAGS)   -c monolish_f.f90 -o test_fort.o
	$(CXX) $(CXXFLAGS) $(LIB_CPU) -c monolish_cpp.cpp -o test_cpp.o
	$(FCC) $(FFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIB_CPU)

gpu:
	$(FCC) $(FFLAGS)   -c monolish_f.f90 -o test_fort.o
	$(CXX) $(CXXFLAGS) $(LIB_GPU) -c monolish_cpp.cpp -o test_cpp.o
	$(FCC) $(FFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIB_GPU)

run_cpu:
	echo "cpu version does not impl."
	# ./test.out
	
run_gpu:
	./test.out
	
clean:
	- rm *.o
	- rm *.out
