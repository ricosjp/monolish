MONOLISH_DIR?=$(HOME)/lib/monolish

CXX=FCC
#CXXFLAGS=-Kfast,openmp,zfill,parallel,simd -std=c++17 -lm -SSL2BLAMP
CXXFLAGS=-Nclang -Ofast -fopenmp -std=c++17 -lm -SSL2BLAMP

LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_a64fx

OBJS=$(FUNC)_$(ARCH).out

all:$(OBJS)

$(FUNC)_$(ARCH).out: $(FUNC).cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(LIBS) 
