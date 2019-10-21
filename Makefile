TARGET=./lib/libmonolish_cpu.so

CXX=g++
CXXFLAGS+=-O3 -std=c++11 -fopenmp -lm -g -Wall -Wno-unused-variable -fPIC 
CXXFLAGS+= -DUSE_AVX2 -mavx2 -mfma
LIBFLAGS=-shared

SRCS =$(notdir $(wildcard  ./src/*.cpp))
SRCS +=$(notdir $(wildcard ./src/equation/*.cpp))
SRCS +=$(notdir $(wildcard ./src/precon/*.cpp))
SRCS +=$(notdir $(wildcard ./src/blas/vector/dot/*.cpp))
SRCS +=$(notdir $(wildcard ./src/blas/or/dot/*.cpp))
SRCS +=$(notdir $(wildcard ./src/external_IF*.cpp))

vpath %.cpp ./src/

vpath %.cpp ./src/equation/

vpath %.cpp ./src/external_IF/

vpath %.cpp ./src/blas/
vpath %.cpp ./src/blas/vector/
vpath %.cpp ./src/blas/vector/dot/
vpath %.cpp ./src/blas/matrix/

OBJDIR=./obj/

OBJS=$(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJS)
	g++ $(CXXFLAGS) $(LIBFLAGS) $(OBJS) -o $(TARGET)

$(OBJDIR)/%.o: %.cpp $(EXTERNALS)
	mkdir -p obj
	g++ $(CXXFLAGS) -I../include -c $< -o $@

$(EXTERNALS):
	git submodule update -i

clean:
	- rm obj/*
	- rm lib/*
