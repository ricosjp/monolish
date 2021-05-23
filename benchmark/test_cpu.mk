MONOLISH_DIR?=$(HOME)/lib/monolish

CXX=g++
CXXFLAGS=-O3 -lm -std=c++17

LIBS=-I $(MONOLISH_DIR)/include/ -L$(MONOLISH_DIR)/lib/ -lmonolish_cpu

OBJS=$(FUNC)_$(ARCH).out

all:$(OBJS)

$(FUNC)_$(ARCH).out: $(FUNC).cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ $(LIBS) 
