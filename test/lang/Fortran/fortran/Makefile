all:
	gfortran -c monolish_f.f90 -o monolish_fort.o
	#g++ -fPIC -c monolish_slu.cpp -o monolish_cpp.o
	g++ -fPIC -c monolish_c.c -o monolish_c.o
	gfortran monolish_fort.o monolish_c.o -lstdc++

clean:
	rm *.o
