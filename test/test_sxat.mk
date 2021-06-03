MONOLISH_DIR?=$(HOME)/lib/monolish

CXX=nc++

BLAS_LIB = -I/opt/nec/ve/nlc/2.3.0/include/ 
BLAS_LIB += -L/opt/nec/ve/nlc/2.3.0/lib/
BLAS_LIB += -llapack -lcblas -lblas_openmp

CXXFLAGS=-O3 -lm -fopenmp -pthread -std=c++17

LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_sxat $(BLAS_LIB) 

OBJS=$(FUNC)_$(ARCH).out

all:$(OBJS)

$(FUNC)_$(ARCH).out: $(FUNC).cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(LIBS) 
