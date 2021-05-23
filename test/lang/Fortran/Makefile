MONOLISH_DIR?=$(HOME)/lib/monolish

FCC ?= gfortran
FFLAGS=-O3 -fopenmp

FXFCC ?= frt
FXFFLAGS=-Kfast,openmp,zfill,parallel,simd -SSL2BLAMP

CXX ?= g++
CXXFLAGS=-O3 -lm -std=c++17 -fopenmp -fPIC

FXCXX ?= FCC
FXCXXFLAGS=-Kfast,openmp,zfill,parallel,simd -std=c++17 -lm -SSL2BLAMP

LIB_CPU=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu
LIB_GPU=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu
LIB_FX=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_a64fx

cpu:
	$(FCC) $(FFLAGS) -c monolish_f.f90 -o test_fort.o
	$(CXX) $(CXXFLAGS) $(LIB_CPU) -c monolish_cpp.cpp -o test_cpp.o
	$(FCC) $(FFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIB_CPU)

gpu:
	$(FCC) $(FFLAGS) -c monolish_f.f90 -o test_fort.o
	$(CXX) $(CXXFLAGS) $(LIB_GPU) -c monolish_cpp.cpp -o test_cpp.o
	$(FCC) $(FFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIB_GPU)

a64fx:
	$(FXFCC) $(FXFFLAGS) -c monolish_f.f90 -o test_fort.o
	$(FXCXX) $(FXCXXFLAGS) $(LIB_FX) -c monolish_cpp.cpp -o test_cpp.o 
	$(FXCXX) $(FXCXXFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIB_FX) 

run_cpu:
#	 ./test.out
	
run_gpu:
	./test.out

run_a64fx:
	./test.out
	
clean:
	- rm *.o
	- rm *.out
