MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/

CXX=g++
CXXFLAGS=-O3

all: cpu 

cpu: innerproduct_cpu.out run_cpu
	
innerproduct_cpu.out: innerproduct.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu 

run_cpu:
	echo "run CPU"
	./innerproduct_cpu.out

clean:
	rm *.out
