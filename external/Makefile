BLASOBJ=/obj/openblas/libopenblas.a

all:BLASOBJ
	echo gomagoma

LAPACKOBJ:$(BLASOBJ)
	mkdir -p obj/scalapack
	cp ./SLmake.inc.example scalapack/SLmake.inc
	cd scalapack; make -j
	cp scalapack/libscalapack.a obj/scalapack


BLASOBJ: 
	mkdir -p obj/openblas
	cd ./OpenBLAS; mkdir -p build; cd build; cmake ..; make -j; 
	cp OpenBLAS/*.h ./obj/openblas/
	cp OpenBLAS/build/lib/* ./obj/openblas/

