MONOLISH_DIR?=$(HOME)/lib/monolish
LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/

CXX=g++
CXXFLAGS=-O3

all: cpu

cpu: cg_cpu.out run_cpu
	
cg_cpu.out: cg.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS) -lmonolish_cpu 

run_cpu:
	echo "run CPU"
	./cg_cpu.out

clean:
	rm *.out
