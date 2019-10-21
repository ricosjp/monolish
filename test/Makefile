CXXFLAGS+=-O3 -std=c++11 -fopenmp -lm -g -Wall -Wno-unused-variable -fPIC 
SRCS=./test.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRCS) -L../lib -I../include/ -lmonolish_cpu

run: 
	 export LD_LIBRARY_PATH=/home/hishinuma/Work/monolish_prototype/lib/:$(LD_LIBRARY_PATH); ./a.out
