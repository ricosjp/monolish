blas_obj=./obj/openblas/libopenblas.a
scalapack_obj=./obj/scalapack/libscalapack.a

all: mumps_obj

mumps_obj: scalapack_obj
	mkdir -p obj/mumps
	mkdir -p obj/mumps/lib
	mkdir -p obj/mumps/include
	cp ./mumps_makefile mumps/Makefile.inc
	make -C mumps -j
	cp mumps/lib/* obj/mumps/lib/
	cp mumps/include/* obj/mumps/include/
	cp mumps/libseq/mpi.h obj/mumps/include/
	cd mumps; make clean

scalapack_obj:
	mkdir -p obj/scalapack
	mkdir -p build
	cd build &&\
		cmake \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_STATIC_LIBS=ON \
		-DCMAKE_INSTALL_PREFIX=./lib/\
		-DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
		../scalapack 
	make -j -C build
	cp build/lib/libscalapack.a obj/scalapack/
	rm -rf build


#allgebra has openblas
# blas_obj:
# 	mkdir -p obj/openblas
# 	cd ./OpenBLAS; mkdir -p build; cd build; cmake ..; make -j; 
# 	cp OpenBLAS/*.h ./obj/openblas/
# 	cp OpenBLAS/build/*.h ./obj/openblas/
# 	cp OpenBLAS/build/lib/* ./obj/openblas/
