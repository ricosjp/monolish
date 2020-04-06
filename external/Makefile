blas_obj=./obj/openblas/libopenblas.a
scalapack_obj=./obj/scalapack/libscalapack.a

all: scalapack_obj

# mumps_obj: scalapack_obj
# 	mkdir -p obj/mumps
# 	mkdir -p mumps/lib
# 	cp ./mumps_makefile mumps/Makefile.inc
# 	cd mumps; make -j
# 	cp mumps/lib/* obj/mumps/
# 	cp mumps/include/* obj/mumps/
#
scalapack_obj: 
	mkdir -p obj/scalapack
	mkdir -p build
	cd build &&\
	pwd && \
	cmake \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_STATIC_LIBS=ON \
		-DCMAKE_INSTALL_PREFIX=${LIB_ROOT} \
		-DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
		-DBLAS_LIBRARIES=$LIB_ROOT/lib/libopenblas.a \
		-DLAPACK_LIBRARIES=$LIB_ROOT/lib/libopenblas.a \
		../scalapack
	make -j


#allgebra has openblas
# blas_obj:
# 	mkdir -p obj/openblas
# 	cd ./OpenBLAS; mkdir -p build; cd build; cmake ..; make -j; 
# 	cp OpenBLAS/*.h ./obj/openblas/
# 	cp OpenBLAS/build/*.h ./obj/openblas/
# 	cp OpenBLAS/build/lib/* ./obj/openblas/
