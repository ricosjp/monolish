MONOLISH_DIR?=$(HOME)/lib/monolish

FCC ?= gfortran
FFLAGS=-O3 -fopenmp

CXX ?= g++
CXXFLAGS=-O3 -lm -std=c++17 -fopenmp

LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

LDFLAGS = -lstdc++ -lmonolish

all:
	$(FCC) $(FFLAGS)   -c monolish_f.f90 -o test_fort.o
	$(CXX) $(CXXFLAGS) $(LIBS) -c monolish_cpp.cpp -o test_cpp.o
	$(FCC) $(FFLAGS) test_fort.o test_cpp.o -lstdc++ -o test.out $(LIBS)
	
clean:
	- rm *.o
	- rm *.out
