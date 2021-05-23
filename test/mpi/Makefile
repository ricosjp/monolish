MONOLISH_DIR?=$(HOME)/lib/monolish

MPICXX=mpic++
CXX=clang++
CXXFLAGS=-O3 -std=c++17
CXXFLAGS+=-fopenmp

CPU_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu
GPU_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu
CPU_MPI_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu_mpi
GPU_MPI_LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_gpu_mpi

#OBJS=$(FUNC)_$(ARCH)_MPI.out

.PHONY: all cpu_mpi cpu clean

all: cpu cpu_mpi gpu gpu_mpi

mpi: cpu_mpi gpu_mpi

cpu:
	$(CXX) $(CXXFLAGS) $(GPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

cpu_mpi:
	$(MPICXX) $(CXXFLAGS) $(CPU_MPI_LIBS) mpi_common.cpp -o mpi_common_$@.o $(LIBS) 
	$(MPICXX) $(CXXFLAGS) $(CPU_MPI_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

gpu:
	$(CXX) $(CXXFLAGS) $(GPU_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 

gpu_mpi:
	$(MPICXX) $(CXXFLAGS) $(GPU_MPI_LIBS) mpi_common.cpp -o mpi_common_$@.o $(LIBS) 
	$(MPICXX) $(CXXFLAGS) $(GPU_MPI_LIBS) vector_blas.cpp -o vector_blas_$@.o $(LIBS) 


run:
	mpirun -np 2 --allow-run-as-root ./mpi_common_cpu_mpi.o 100000
	mpirun -np 2 --allow-run-as-root ./mpi_common_gpu_mpi.o 100000
	./vector_blas_cpu.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_cpu_mpi.o 100000
	./vector_blas_gpu.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_gpu_mpi.o 100000

run_mpi:
	mpirun -np 2 --allow-run-as-root ./mpi_common_cpu_mpi.o 100000
	mpirun -np 2 --allow-run-as-root ./mpi_common_gpu_mpi.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_cpu_mpi.o 100000
	mpirun -np 2 --allow-run-as-root ./vector_blas_gpu_mpi.o 100000

clean:
	- rm *.o *.out


# $(FUNC)_$(ARCH).out: $(FUNC).cpp
# 	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(LIBS) 
