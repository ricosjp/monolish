blas_obj=./obj/openblas/libopenblas.a
scalapack_obj=./obj/scalapack/libscalapack.a

all: blas_obj

# mumps_obj: scalapack_obj
# 	mkdir -p obj/mumps
# 	mkdir -p mumps/lib
# 	cp ./mumps_makefile mumps/Makefile.inc
# 	cd mumps; make -j
# 	cp mumps/lib/* obj/mumps/
# 	cp mumps/include/* obj/mumps/
#
# scalapack_obj: blas_obj
# 	mkdir -p obj/scalapack
# 	cp ./SLmake.inc.example scalapack/SLmake.inc
# 	cd scalapack; make -j
# 	cp scalapack/libscalapack.a obj/scalapack

blas_obj:
	mkdir -p obj/openblas
	cd ./OpenBLAS; mkdir -p build; cd build; cmake ..; make -j; 
	cp OpenBLAS/*.h ./obj/openblas/
	cp OpenBLAS/build/*.h ./obj/openblas/
	cp OpenBLAS/build/lib/* ./obj/openblas/

