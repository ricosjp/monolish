CXX=g++
CXXFLAGS+=-O3 -std=c++11 -fopenmp -lm -g -Wall -Wno-unused-variable
LIBS=-I ../include -L ../lib -lmonolish_cpu
MONOLISH=$(HOME)/Work/monolish/lib/# gomi!!

OBJS=cg.out blas.out lu.cpp

all:$(OBJS)

cg.out: cg.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@
	export LD_LIBRARY_PATH=$(MONOLISH):$(LD_LIBRARY_PATH); ./$@
	
blas.out: blas.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@
	export LD_LIBRARY_PATH=$(MONOLISH):$(LD_LIBRARY_PATH); ./$@
