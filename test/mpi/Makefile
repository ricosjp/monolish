MONOLISH_DIR?=$(HOME)/lib/monolish

MPICXX=mpic++
CXX=clang++
CXXFLAGS=-O3 -std=c++14
CXXFLAGS+=-fopenmp

CPU_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu
GPU_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu

#OBJS=$(FUNC)_$(ARCH)_MPI.out

.PHONY: all cpu_mpi cpu clean

all: cpu cpu_mpi gpu gpu_mpi

cpu:
	$(CXX) $(CXXFLAGS) $(GPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

cpu_mpi:
	$(MPICXX) $(CXXFLAGS) $(CPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

gpu:
	$(CXX) $(CXXFLAGS) $(GPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

gpu_mpi:
	$(MPICXX) $(CXXFLAGS) $(GPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 


run:
	./vector_blas_cpu.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_cpu.o 100000
	./vector_blas_gpu.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_gpu.o 100000

clean:
	- rm *.o *.out


# $(FUNC)_$(ARCH).out: $(FUNC).cpp
# 	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(LIBS) 
