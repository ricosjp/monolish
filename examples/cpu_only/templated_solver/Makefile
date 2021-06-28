MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/

CXX=g++
CXXFLAGS=-O3

all: cpu

cpu: solver_cpu.out run_cpu
	
solver_cpu.out: solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu 

run_cpu:
	echo "run CPU"
	./solver_cpu.out

clean:
	rm *.out
