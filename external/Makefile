BLASOBJ=/obj/openblas/libopenblas.a

all:BLASOBJ
	echo gomagoma

LAPACKOBJ

BLASOBJ: 
	mkdir -p obj/openblas
	cd ./OpenBLAS; mkdir -p build; cd build; cmake ..; make -j; 
	cp OpenBLAS/*.h ./obj/openblas/
	cp OpenBLAS/build/lib/* ./obj/openblas/

